//
// RT64
//
// OpenVR (SteamVR) implementation of the VR provider.
//

#include "hle/rt64_vr.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <sstream>

#include <openvr.h>

#include "render/rt64_render_worker.h"
#include "shared/rt64_hlsl.h"
#include "vulkan/rt64_vulkan.h"

#ifdef _WIN64
#   include "d3d12/rt64_d3d12.h"
#endif

namespace RT64 {
    // OpenVR matrices use column vectors (p' = M * p) and store the translation in the last column. RT64 uses row vectors.
    static hlslpp::float4x4 fromOpenVR(const vr::HmdMatrix34_t &m) {
        interop::float4x4 r = interop::float4x4::identity();
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                r[i][j] = m.m[j][i];
            }

            r[3][i] = m.m[i][3];
        }

        return r;
    }

    static std::vector<std::string> splitExtensions(const std::string &extensions) {
        std::vector<std::string> result;
        std::istringstream stream(extensions);
        std::string extension;
        while (stream >> extension) {
            result.emplace_back(extension);
        }

        return result;
    }

    struct OpenVRProvider : VRProvider {
        vr::IVRSystem *system = nullptr;
        vr::IVRCompositor *compositor = nullptr;
        RenderDevice *device = nullptr;
        UserConfiguration::GraphicsAPI graphicsAPI = UserConfiguration::GraphicsAPI::OptionCount;
        std::atomic<bool> active = false;
        std::atomic<bool> recenterRequested = true;
        uint32_t refreshRate = 90;
        uint32_t recommendedWidth = 1440;
        uint32_t recommendedHeight = 1600;
        VREyePose eyePoses[2];
        vr::TrackedDevicePose_t trackedPoses[vr::k_unMaxTrackedDeviceCount] = {};
        interop::float4x4 trackingToAnchor = interop::float4x4::identity();
        hlslpp::float4x4 lastHeadToAnchor = hlslpp::float4x4::identity();
        bool lastHeadValid = false;
        bool submitErrorReported = false;

        ~OpenVRProvider() override {
            if (system != nullptr) {
                vr::VR_Shutdown();
                system = nullptr;
                compositor = nullptr;
            }
        }

        bool initialize() {
            if (!vr::VR_IsRuntimeInstalled()) {
                fprintf(stderr, "OpenVR: SteamVR is not installed.\n");
                return false;
            }

            if (!vr::VR_IsHmdPresent()) {
                fprintf(stderr, "OpenVR: No headset was detected.\n");
                return false;
            }

            vr::EVRInitError initError = vr::VRInitError_None;
            system = vr::VR_Init(&initError, vr::VRApplication_Scene);
            if (initError != vr::VRInitError_None) {
                fprintf(stderr, "OpenVR: Unable to initialize the runtime: %s\n", vr::VR_GetVRInitErrorAsEnglishDescription(initError));
                system = nullptr;
                return false;
            }

            compositor = vr::VRCompositor();
            if (compositor == nullptr) {
                fprintf(stderr, "OpenVR: Unable to retrieve the compositor.\n");
                return false;
            }

            // Seated tracking: the origin is a point the player recenters on, which is anchored to the game's camera.
            compositor->SetTrackingSpace(vr::TrackingUniverseSeated);

            system->GetRecommendedRenderTargetSize(&recommendedWidth, &recommendedHeight);

            vr::ETrackedPropertyError propertyError = vr::TrackedProp_Success;
            const float displayFrequency = system->GetFloatTrackedDeviceProperty(vr::k_unTrackedDeviceIndex_Hmd, vr::Prop_DisplayFrequency_Float, &propertyError);
            if ((propertyError == vr::TrackedProp_Success) && (displayFrequency > 1.0f)) {
                refreshRate = uint32_t(lroundf(displayFrequency));
            }

            updateEyes();

            fprintf(stdout, "OpenVR: Initialized. Recommended eye size %ux%u at %u Hz.\n", recommendedWidth, recommendedHeight, refreshRate);
            return true;
        }

        void updateEyes() {
            for (uint32_t e = 0; e < 2; e++) {
                const vr::EVREye eye = (e == 0) ? vr::Eye_Left : vr::Eye_Right;
                float left, right, top, bottom;
                system->GetProjectionRaw(eye, &left, &right, &top, &bottom);

                // OpenVR's top and bottom values are expressed with Y pointing down. Convert them to the Y up convention.
                VREyePose &eyePose = eyePoses[e];
                eyePose.tanLeft = left;
                eyePose.tanRight = right;
                eyePose.tanUp = bottom;
                eyePose.tanDown = top;
                eyePose.eyeToHead = fromOpenVR(system->GetEyeToHeadTransform(eye));
            }
        }

        void pollEvents() {
            vr::VREvent_t event;
            while (system->PollNextEvent(&event, sizeof(event))) {
                switch (event.eventType) {
                case vr::VREvent_IpdChanged:
                    updateEyes();
                    break;
                case vr::VREvent_Quit:
                    // SteamVR is shutting down. Stop sending frames and let the game keep running on the window.
                    system->AcknowledgeQuit_Exiting();
                    active = false;
                    break;
                default:
                    break;
                }
            }
        }

        void recenter(const interop::float4x4 &headToTracking) {
            // Keep only the heading of the head so the horizon stays level, but use its full position.
            float back[3] = { headToTracking[2][0], 0.0f, headToTracking[2][2] };
            float length = sqrtf(back[0] * back[0] + back[2] * back[2]);
            if (length < 1e-4f) {
                back[0] = 0.0f;
                back[2] = 1.0f;
                length = 1.0f;
            }

            back[0] /= length;
            back[2] /= length;

            // Rows hold the axes of the anchor in tracking space: right = up x back.
            interop::float4x4 anchorToTracking = interop::float4x4::identity();
            anchorToTracking[0][0] = back[2];
            anchorToTracking[0][1] = 0.0f;
            anchorToTracking[0][2] = -back[0];
            anchorToTracking[1][0] = 0.0f;
            anchorToTracking[1][1] = 1.0f;
            anchorToTracking[1][2] = 0.0f;
            anchorToTracking[2][0] = back[0];
            anchorToTracking[2][1] = 0.0f;
            anchorToTracking[2][2] = back[2];
            anchorToTracking[3][0] = headToTracking[3][0];
            anchorToTracking[3][1] = headToTracking[3][1];
            anchorToTracking[3][2] = headToTracking[3][2];

            // Rigid inverse.
            interop::float4x4 anchorInverse = interop::float4x4::identity();
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    anchorInverse[i][j] = anchorToTracking[j][i];
                }
            }

            for (int j = 0; j < 3; j++) {
                anchorInverse[3][j] = -(anchorToTracking[3][0] * anchorInverse[0][j] + anchorToTracking[3][1] * anchorInverse[1][j] + anchorToTracking[3][2] * anchorInverse[2][j]);
            }

            trackingToAnchor = anchorInverse;
        }

        // VRProvider

        std::vector<std::string> getVulkanInstanceExtensions() override {
            const uint32_t size = compositor->GetVulkanInstanceExtensionsRequired(nullptr, 0);
            if (size == 0) {
                return {};
            }

            std::string extensions(size, '\0');
            compositor->GetVulkanInstanceExtensionsRequired(extensions.data(), size);
            extensions.resize(strnlen(extensions.c_str(), size));
            return splitExtensions(extensions);
        }

        std::vector<std::string> getVulkanDeviceExtensions(void *vkPhysicalDevice) override {
            VkPhysicalDevice_T *physicalDevice = reinterpret_cast<VkPhysicalDevice_T *>(vkPhysicalDevice);
            const uint32_t size = compositor->GetVulkanDeviceExtensionsRequired(physicalDevice, nullptr, 0);
            if (size == 0) {
                return {};
            }

            std::string extensions(size, '\0');
            compositor->GetVulkanDeviceExtensionsRequired(physicalDevice, extensions.data(), size);
            extensions.resize(strnlen(extensions.c_str(), size));
            return splitExtensions(extensions);
        }

        bool setupGraphics(RenderInterface *renderInterface, RenderDevice *device, UserConfiguration::GraphicsAPI graphicsAPI) override {
            (void)(renderInterface);

            this->device = device;
            this->graphicsAPI = graphicsAPI;
            switch (graphicsAPI) {
            case UserConfiguration::GraphicsAPI::Vulkan:
                // The queue used for rendering can be shared with other threads, so the runtime must never access it on its own
                // during WaitGetPoses. The application performs the handoff explicitly while holding the queue's lock instead.
                compositor->SetExplicitTimingMode(vr::VRCompositorTimingMode_Explicit_ApplicationPerformsPostPresentHandoff);
                break;
#       ifdef _WIN64
            case UserConfiguration::GraphicsAPI::D3D12:
                break;
#       endif
            default:
                fprintf(stderr, "OpenVR: The selected graphics API is not supported. Use D3D12 or Vulkan.\n");
                return false;
            }

            active = true;
            return true;
        }

        bool isActive() const override {
            return active;
        }

        uint32_t getRefreshRate() const override {
            return refreshRate;
        }

        void getRecommendedRenderSize(uint32_t &width, uint32_t &height) const override {
            width = recommendedWidth;
            height = recommendedHeight;
        }

        std::mutex *getVulkanQueueMutex(RenderWorker *worker) {
            VulkanCommandQueue *commandQueue = static_cast<VulkanCommandQueue *>(worker->commandQueue.get());
            return commandQueue->queue->mutex.get();
        }

        bool waitFrame(RenderWorker *worker, VRFrame &frame) override {
            pollEvents();

            const vr::EVRCompositorError waitError = compositor->WaitGetPoses(trackedPoses, vr::k_unMaxTrackedDeviceCount, nullptr, 0);

            // With explicit timing, the runtime must be notified right before the frame's work starts being submitted.
            if (graphicsAPI == UserConfiguration::GraphicsAPI::Vulkan) {
                std::scoped_lock<std::mutex> queueLock(*getVulkanQueueMutex(worker));
                compositor->SubmitExplicitTimingData();
            }
            const vr::TrackedDevicePose_t &hmdPose = trackedPoses[vr::k_unTrackedDeviceIndex_Hmd];
            if ((waitError == vr::VRCompositorError_None) && hmdPose.bPoseIsValid) {
                const interop::float4x4 headToTracking = fromOpenVR(hmdPose.mDeviceToAbsoluteTracking);
                if (recenterRequested.exchange(false)) {
                    recenter(headToTracking);
                }

                hlslpp::float4x4 headToTrackingMatrix = headToTracking;
                hlslpp::float4x4 trackingToAnchorMatrix = trackingToAnchor;
                lastHeadToAnchor = hlslpp::mul(headToTrackingMatrix, trackingToAnchorMatrix);
                lastHeadValid = true;
            }

            frame.poseValid = lastHeadValid;
            frame.headToTracking = lastHeadToAnchor;
            frame.eyes[0] = eyePoses[0];
            frame.eyes[1] = eyePoses[1];
            return (waitError == vr::VRCompositorError_None);
        }

        void submitEye(RenderWorker *worker, uint32_t eyeIndex, const VRSubmitTexture &submitTexture) override {
            if (submitTexture.texture == nullptr) {
                return;
            }

            const vr::EVREye eye = (eyeIndex == 0) ? vr::Eye_Left : vr::Eye_Right;
            vr::VRTextureBounds_t bounds;
            bounds.uMin = submitTexture.uMin;
            bounds.vMin = submitTexture.vMin;
            bounds.uMax = submitTexture.uMax;
            bounds.vMax = submitTexture.vMax;

            vr::Texture_t texture = {};
            texture.eColorSpace = vr::ColorSpace_Gamma;

            vr::EVRCompositorError submitError = vr::VRCompositorError_None;
            if (graphicsAPI == UserConfiguration::GraphicsAPI::Vulkan) {
                // The runtime copies the image with a transfer operation on the provided queue.
                {
                    RenderWorkerExecution execution(worker);
                    worker->commandList->barriers(RenderBarrierStage::COPY, RenderTextureBarrier(submitTexture.texture, RenderTextureLayout::COPY_SOURCE));
                }

                VulkanDevice *vulkanDevice = static_cast<VulkanDevice *>(device);
                VulkanTexture *vulkanTexture = static_cast<VulkanTexture *>(submitTexture.texture);
                VulkanCommandQueue *commandQueue = static_cast<VulkanCommandQueue *>(worker->commandQueue.get());
                vr::VRVulkanTextureData_t textureData = {};
                textureData.m_nImage = uint64_t(vulkanTexture->vk);
                textureData.m_pDevice = vulkanDevice->vk;
                textureData.m_pPhysicalDevice = vulkanDevice->physicalDevice;
                textureData.m_pInstance = vulkanDevice->renderInterface->instance;
                textureData.m_pQueue = commandQueue->queue->vk;
                textureData.m_nQueueFamilyIndex = commandQueue->familyIndex;
                textureData.m_nWidth = submitTexture.width;
                textureData.m_nHeight = submitTexture.height;
                textureData.m_nFormat = uint32_t(vulkanTexture->imageFormat);
                textureData.m_nSampleCount = 1;
                texture.handle = &textureData;
                texture.eType = vr::TextureType_Vulkan;

                std::scoped_lock<std::mutex> queueLock(*commandQueue->queue->mutex);
                submitError = compositor->Submit(eye, &texture, &bounds, vr::Submit_Default);
            }
#       ifdef _WIN64
            else if (graphicsAPI == UserConfiguration::GraphicsAPI::D3D12) {
                // The runtime expects the resource to be in the pixel shader resource state.
                {
                    RenderWorkerExecution execution(worker);
                    worker->commandList->barriers(RenderBarrierStage::GRAPHICS, RenderTextureBarrier(submitTexture.texture, RenderTextureLayout::SHADER_READ));
                }

                D3D12Texture *d3d12Texture = static_cast<D3D12Texture *>(submitTexture.texture);
                D3D12CommandQueue *commandQueue = static_cast<D3D12CommandQueue *>(worker->commandQueue.get());
                vr::D3D12TextureData_t textureData = {};
                textureData.m_pResource = d3d12Texture->d3d;
                textureData.m_pCommandQueue = commandQueue->d3d;
                textureData.m_nNodeMask = 0;
                texture.handle = &textureData;
                texture.eType = vr::TextureType_DirectX12;
                submitError = compositor->Submit(eye, &texture, &bounds, vr::Submit_Default);
            }
#       endif

            if ((submitError != vr::VRCompositorError_None) && (submitError != vr::VRCompositorError_DoNotHaveFocus) && !submitErrorReported) {
                fprintf(stderr, "OpenVR: Submitting the eye texture failed with error %d.\n", int(submitError));
                submitErrorReported = true;
            }
        }

        void endFrame(RenderWorker *worker) override {
            if (graphicsAPI == UserConfiguration::GraphicsAPI::Vulkan) {
                std::scoped_lock<std::mutex> queueLock(*getVulkanQueueMutex(worker));
                compositor->PostPresentHandoff();
            }
            else {
                compositor->PostPresentHandoff();
            }
        }

        void requestRecenter() override {
            recenterRequested = true;
        }
    };

    std::unique_ptr<VRProvider> CreateOpenVRProvider() {
        std::unique_ptr<OpenVRProvider> provider = std::make_unique<OpenVRProvider>();
        if (!provider->initialize()) {
            return nullptr;
        }

        return provider;
    }
};
