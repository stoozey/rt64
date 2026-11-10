//
// RT64
//
// Stereo (VR) rendering support.
//
// The VR mode keeps the game's own camera as the "anchor" of the player's head: every perspective projection tagged
// with the configured world camera matrix group ID is rendered twice per displayed frame, once for each eye, with the
// tracked head pose and the headset's per-eye projection applied on top of the game's view matrix. Everything else
// (HUD, text boxes, menus) is remapped onto a head-locked virtual panel, while full screen effects (fades, clears,
// transitions) are left covering the entire eye.
//
// The runtime that provides the poses and receives the rendered images is abstracted behind VRProvider so the renderer
// doesn't depend on any particular VR SDK.
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "common/rt64_math.h"
#include "common/rt64_user_configuration.h"
#include "rhi/rt64_render_interface.h"

namespace RT64 {
    struct DrawData;
    struct RenderWorker;

    struct VRConfiguration {
        // Whether VR should be attempted at all. Must be set before the application is set up, as some graphics APIs
        // require extensions to be enabled when the device is created.
        bool enabled = false;

        // Matrix group ID (as used by the extended GBI matrix group commands) that identifies the game's main world
        // camera projection. Only projections tagged with this ID receive head tracking and stereo separation.
        uint32_t worldCameraMatrixId = 0;

        // Amount of game units that correspond to one real world meter. Controls how big the world feels and how much
        // the camera moves when the player moves their head.
        float unitsPerMeter = 30.0f;

        // Distance in meters and horizontal angular size in degrees of the head-locked panel that HUD elements, text
        // boxes and menus are drawn on.
        float hudDistance = 1.5f;
        float hudHorizontalFov = 55.0f;

        // Remove the pitch and roll from the game camera so the horizon always stays level. Only the real head
        // rotation of the player tilts the view, which is generally more comfortable.
        bool levelHorizon = false;

        // Multiplier applied to the resolution recommended by the VR runtime.
        float renderScale = 1.0f;
    };

    // All matrices use the same row vector convention as the rest of RT64 (p' = p * M) and are expressed in meters.
    struct VREyePose {
        hlslpp::float4x4 eyeToHead = hlslpp::float4x4::identity();

        // Tangents of the half angles of the eye's frustum in a Y up space. Left and down are negative.
        float tanLeft = -1.0f;
        float tanRight = 1.0f;
        float tanUp = 1.0f;
        float tanDown = -1.0f;
    };

    struct VRFrame {
        bool poseValid = false;
        hlslpp::float4x4 headToTracking = hlslpp::float4x4::identity();
        VREyePose eyes[2];
    };

    struct VRSubmitTexture {
        RenderTexture *texture = nullptr;
        RenderFormat format = RenderFormat::UNKNOWN;
        uint32_t width = 0;
        uint32_t height = 0;

        // Region of the texture that holds the eye's image, in normalized texture coordinates.
        float uMin = 0.0f;
        float vMin = 0.0f;
        float uMax = 1.0f;
        float vMax = 1.0f;
    };

    struct VRProvider {
        virtual ~VRProvider() { }

        // Extensions required by the runtime when the Vulkan backend is in use. The instance extensions are queried before
        // the instance is created, while the device extensions are queried for the physical device that is about to be used.
        virtual std::vector<std::string> getVulkanInstanceExtensions() = 0;
        virtual std::vector<std::string> getVulkanDeviceExtensions(void *vkPhysicalDevice) = 0;

        // Called once the graphics device has been created. Returns false if the runtime can't work with the device.
        virtual bool setupGraphics(RenderInterface *renderInterface, RenderDevice *device, UserConfiguration::GraphicsAPI graphicsAPI) = 0;

        // Whether the runtime is ready to receive frames.
        virtual bool isActive() const = 0;

        // Refresh rate of the headset in Hz.
        virtual uint32_t getRefreshRate() const = 0;

        // Per-eye resolution suggested by the runtime.
        virtual void getRecommendedRenderSize(uint32_t &width, uint32_t &height) const = 0;

        // Blocks until the runtime wants a new frame to be rendered and retrieves the poses to render it with. The worker
        // is the one that will render and submit the frame.
        virtual bool waitFrame(RenderWorker *worker, VRFrame &frame) = 0;

        // Hands the rendered image of one eye over to the runtime. The texture must not be multisampled. Any work that
        // renders the texture must've been executed on the worker's queue before calling this function.
        virtual void submitEye(RenderWorker *worker, uint32_t eyeIndex, const VRSubmitTexture &texture) = 0;

        // Called after both eyes were submitted.
        virtual void endFrame(RenderWorker *worker) = 0;

        // Sets the current head position and heading as the new origin the game camera is anchored to.
        virtual void requestRecenter() = 0;
    };

    // Everything the renderer needs to know to produce the image for one eye.
    struct VRRenderEye {
        uint32_t eyeIndex = 0;
        uint32_t worldCameraMatrixId = 0;
        bool levelHorizon = false;

        // Transform from the game camera's space to the eye's space, in game units.
        hlslpp::float4x4 cameraToEye = hlslpp::float4x4::identity();
        float tanLeft = -1.0f;
        float tanRight = 1.0f;
        float tanUp = 1.0f;
        float tanDown = -1.0f;

        // Color image (RDRAM address) that is being sent to the headset. Only the framebuffer pairs that draw to this image
        // get their HUD elements remapped onto the virtual panel.
        uint32_t targetAddress = 0;

        // Native size of the region of the color image that is shown in the headset.
        float regionWidth = 320.0f;
        float regionHeight = 240.0f;

        // Rectangle of the eye's image the virtual panel covers, in normalized coordinates (Y down).
        float hudLeft = 0.25f;
        float hudTop = 0.25f;
        float hudRight = 0.75f;
        float hudBottom = 0.75f;

        // Fills all the fields of the structure from the configuration and the frame provided by the runtime.
        void setup(const VRConfiguration &config, const VRFrame &frame, uint32_t eyeIndex, uint32_t targetAddress, float regionWidth, float regionHeight);

        // Returns whether the transforms index of the draw data belongs to the game's world camera.
        bool isWorldTransform(const DrawData &drawData, uint32_t transformsIndex) const;

        // Modifies the view and projection matrices of the world camera so they render the eye's view.
        void applyToWorldCamera(hlslpp::float4x4 &viewMatrix, hlslpp::float4x4 &projMatrix, const hlslpp::float3 &viewportScale, const hlslpp::float3 &viewportTranslate) const;

        // Maps a rectangle in target pixel coordinates onto the virtual panel.
        void mapToHUD(float &x, float &y, float &width, float &height, hlslpp::float2 resolutionScale) const;
        void mapToHUD(int32_t &left, int32_t &top, int32_t &right, int32_t &bottom, hlslpp::float2 resolutionScale) const;
    };

    // Creates the provider for the OpenVR (SteamVR) runtime. Returns nullptr if support wasn't built in or if the runtime
    // couldn't be initialized.
    std::unique_ptr<VRProvider> CreateOpenVRProvider();

    // Removes the pitch and roll of a view matrix while keeping its position and heading.
    hlslpp::float4x4 levelViewMatrix(const hlslpp::float4x4 &viewMatrix);
};
