//
// RT64
//

#include "rt64_vr.h"

#include <algorithm>
#include <cmath>

#include "rt64_workload.h"

namespace RT64 {
    // Small helpers that operate on plain matrices to keep the element access simple. Row vector convention (p' = p * M).

    typedef interop::float4x4 Mat4;

    static Mat4 matMul(const Mat4 &a, const Mat4 &b) {
        Mat4 r;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] + a[i][3] * b[3][j];
            }
        }

        return r;
    }

    // Inverse of a matrix that is only composed of a rotation and a translation.
    static Mat4 rigidInverse(const Mat4 &m) {
        Mat4 r = Mat4::identity();
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                r[i][j] = m[j][i];
            }
        }

        for (int j = 0; j < 3; j++) {
            r[3][j] = -(m[3][0] * r[0][j] + m[3][1] * r[1][j] + m[3][2] * r[2][j]);
        }

        return r;
    }

    static void transformPoint(const Mat4 &m, const float p[3], float out[3]) {
        for (int j = 0; j < 3; j++) {
            out[j] = p[0] * m[0][j] + p[1] * m[1][j] + p[2] * m[2][j] + m[3][j];
        }
    }

    static void normalize3(float v[3]) {
        const float length = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        if (length > 1e-8f) {
            v[0] /= length;
            v[1] /= length;
            v[2] /= length;
        }
    }

    // VRRenderEye

    void VRRenderEye::setup(const VRConfiguration &config, const VRFrame &frame, uint32_t eyeIndex, uint32_t targetAddress, float regionWidth, float regionHeight) {
        const VREyePose &eyePose = frame.eyes[eyeIndex];
        this->eyeIndex = eyeIndex;
        this->worldCameraMatrixId = config.worldCameraMatrixId;
        this->levelHorizon = config.levelHorizon;
        this->targetAddress = targetAddress;
        this->regionWidth = regionWidth;
        this->regionHeight = regionHeight;
        tanLeft = eyePose.tanLeft;
        tanRight = eyePose.tanRight;
        tanUp = eyePose.tanUp;
        tanDown = eyePose.tanDown;

        // The game camera is anchored to the origin of the tracking space. A point in the camera space must go through the
        // inverse of the head pose and then through the inverse of the eye pose to end up in the eye space. The rotations
        // don't depend on the scale, so the scale only needs to be applied to the translation of the final transform.
        const Mat4 headToTracking = frame.poseValid ? Mat4(frame.headToTracking) : Mat4::identity();
        const Mat4 eyeToHead = Mat4(eyePose.eyeToHead);
        Mat4 trackingToEye = matMul(rigidInverse(headToTracking), rigidInverse(eyeToHead));
        const float unitsPerMeter = std::max(config.unitsPerMeter, 1e-3f);
        trackingToEye[3][0] *= unitsPerMeter;
        trackingToEye[3][1] *= unitsPerMeter;
        trackingToEye[3][2] *= unitsPerMeter;
        cameraToEye = trackingToEye;

        // Place the virtual panel in front of the head, then project its corners into the eye to find the rectangle it covers.
        const float hudDistance = std::max(config.hudDistance, 0.05f);
        const float hudFov = std::clamp(config.hudHorizontalFov, 5.0f, 170.0f) * (3.14159265f / 180.0f);
        const float hudHalfWidth = hudDistance * tanf(hudFov / 2.0f);
        const float hudHalfHeight = hudHalfWidth * (regionHeight / std::max(regionWidth, 1.0f));
        const Mat4 headToEye = rigidInverse(eyeToHead);
        hudLeft = hudTop = 1e9f;
        hudRight = hudBottom = -1e9f;
        for (int c = 0; c < 4; c++) {
            const float headPoint[3] = { (c & 1) ? hudHalfWidth : -hudHalfWidth, (c & 2) ? hudHalfHeight : -hudHalfHeight, -hudDistance };
            float eyePoint[3];
            transformPoint(headToEye, headPoint, eyePoint);

            const float depth = std::max(-eyePoint[2], 1e-3f);
            const float u = eyePoint[0] / depth;
            const float v = eyePoint[1] / depth;
            const float ndcX = (2.0f * u - (tanRight + tanLeft)) / (tanRight - tanLeft);
            const float ndcY = (2.0f * v - (tanUp + tanDown)) / (tanUp - tanDown);
            const float x = (ndcX + 1.0f) / 2.0f;
            const float y = (1.0f - ndcY) / 2.0f;
            hudLeft = std::min(hudLeft, x);
            hudRight = std::max(hudRight, x);
            hudTop = std::min(hudTop, y);
            hudBottom = std::max(hudBottom, y);
        }
    }

    bool VRRenderEye::isWorldTransform(const DrawData &drawData, uint32_t transformsIndex) const {
        if ((transformsIndex >= drawData.viewProjTransformGroups.size()) || (worldCameraMatrixId == 0)) {
            return false;
        }

        const uint32_t groupIndex = drawData.viewProjTransformGroups[transformsIndex];
        if (groupIndex >= drawData.transformGroups.size()) {
            return false;
        }

        return drawData.transformGroups[groupIndex].matrixId == worldCameraMatrixId;
    }

    void VRRenderEye::applyToWorldCamera(hlslpp::float4x4 &viewMatrix, hlslpp::float4x4 &projMatrix, const hlslpp::float3 &viewportScale, const hlslpp::float3 &viewportTranslate) const {
        // Only perspective projections can be turned into the eye's view. Leave anything else untouched.
        Mat4 proj(projMatrix);
        const float wScale = -proj[2][3];
        if (fabsf(wScale) < 1e-8f) {
            return;
        }

        // Anchor the head to the game camera.
        const hlslpp::float4x4 gameView = levelHorizon ? levelViewMatrix(viewMatrix) : viewMatrix;
        viewMatrix = matMul(Mat4(gameView), Mat4(cameraToEye));

        // Replace the horizontal and vertical terms of the game's projection with the eye's frustum while keeping the depth
        // terms intact, as the game relies on its own near and far planes for depth precision and fog.

        const float width = tanRight - tanLeft;
        const float height = tanUp - tanDown;
        proj[0][0] = (2.0f / width) * wScale;
        proj[1][0] = 0.0f;
        proj[2][0] = ((tanRight + tanLeft) / width) * wScale;
        proj[3][0] = 0.0f;
        proj[0][1] = 0.0f;
        proj[1][1] = (2.0f / height) * wScale;
        proj[2][1] = ((tanUp + tanDown) / height) * wScale;
        proj[3][1] = 0.0f;

        // The game's viewport doesn't necessarily cover the entire screen (e.g. letterboxing). Correct the projection so the
        // eye's frustum maps to the full region of the image regardless of the viewport the game used.
        const float sx = viewportScale.x;
        const float sy = viewportScale.y;
        const float tx = viewportTranslate.x;
        const float ty = viewportTranslate.y;
        if ((fabsf(sx) > 1e-6f) && (fabsf(sy) > 1e-6f)) {
            Mat4 correction = Mat4::identity();
            correction[0][0] = regionWidth / (2.0f * sx);
            correction[3][0] = (regionWidth / 2.0f - tx) / sx;
            correction[1][1] = regionHeight / (2.0f * sy);
            correction[3][1] = (ty - regionHeight / 2.0f) / sy;
            proj = matMul(proj, correction);
        }

        projMatrix = proj;
    }

    void VRRenderEye::mapToHUD(float &x, float &y, float &width, float &height, hlslpp::float2 resolutionScale) const {
        const float scaleX = hudRight - hudLeft;
        const float scaleY = hudBottom - hudTop;
        const float offsetX = hudLeft * regionWidth * resolutionScale.x;
        const float offsetY = hudTop * regionHeight * resolutionScale.y;
        x = x * scaleX + offsetX;
        y = y * scaleY + offsetY;
        width *= scaleX;
        height *= scaleY;
    }

    void VRRenderEye::mapToHUD(int32_t &left, int32_t &top, int32_t &right, int32_t &bottom, hlslpp::float2 resolutionScale) const {
        float x = float(left);
        float y = float(top);
        float w = float(right - left);
        float h = float(bottom - top);
        mapToHUD(x, y, w, h, resolutionScale);
        left = int32_t(floorf(x));
        top = int32_t(floorf(y));
        right = int32_t(ceilf(x + w));
        bottom = int32_t(ceilf(y + h));
    }

    hlslpp::float4x4 levelViewMatrix(const hlslpp::float4x4 &viewMatrix) {
        const Mat4 view(viewMatrix);

        // With row vectors, the columns of the rotation hold the camera's axes in world space.
        float right[3] = { view[0][0], view[1][0], view[2][0] };
        float up[3] = { view[0][1], view[1][1], view[2][1] };
        float back[3] = { view[0][2], view[1][2], view[2][2] };
        normalize3(right);
        normalize3(up);
        normalize3(back);

        // Camera position in world space: t = -c * R  =>  c = -t * R^T.
        float position[3];
        for (int j = 0; j < 3; j++) {
            position[j] = -(view[3][0] * view[j][0] + view[3][1] * view[j][1] + view[3][2] * view[j][2]);
        }

        // Find the heading. If the camera is looking straight up or down, use the up vector instead.
        float flatBack[3] = { back[0], 0.0f, back[2] };
        if ((flatBack[0] * flatBack[0] + flatBack[2] * flatBack[2]) < 1e-6f) {
            const float sign = (back[1] > 0.0f) ? 1.0f : -1.0f;
            flatBack[0] = -up[0] * sign;
            flatBack[2] = -up[2] * sign;
        }

        normalize3(flatBack);

        // right = up x back.
        const float levelUp[3] = { 0.0f, 1.0f, 0.0f };
        const float levelRight[3] = {
            levelUp[1] * flatBack[2] - levelUp[2] * flatBack[1],
            levelUp[2] * flatBack[0] - levelUp[0] * flatBack[2],
            levelUp[0] * flatBack[1] - levelUp[1] * flatBack[0]
        };

        Mat4 level = Mat4::identity();
        for (int i = 0; i < 3; i++) {
            level[i][0] = levelRight[i];
            level[i][1] = levelUp[i];
            level[i][2] = flatBack[i];
        }

        for (int j = 0; j < 3; j++) {
            level[3][j] = -(position[0] * level[0][j] + position[1] * level[1][j] + position[2] * level[2][j]);
        }

        return level;
    }

#if !RT64_OPENVR_ENABLED
    std::unique_ptr<VRProvider> CreateOpenVRProvider() {
        return nullptr;
    }
#endif
};
