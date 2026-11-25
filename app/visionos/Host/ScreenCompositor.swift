import ARKit
import CompositorServices
import Metal
import MetalFX
import QuartzCore
import SwiftUI

/// The layer's setup: no foveation, so the views need no rasterization rate
/// maps, and a depth target, which the compositor uses for reprojection.
/// The layout stays the one the system picks; the drawing handles all three.
struct ScreenLayerConfiguration: CompositorLayerConfiguration {
    func makeConfiguration(
        capabilities: LayerRenderer.Capabilities,
        configuration: inout LayerRenderer.Configuration
    ) {
        configuration.isFoveationEnabled = false
        configuration.depthFormat = .depth32Float
        let layouts = capabilities.supportedLayouts(options: [])
        if !layouts.contains(configuration.layout), let first = layouts.first {
            configuration.layout = first
        }
    }
}

/// Shows the game's latest frame on a world-locked virtual screen.
///
/// The loop runs on a thread of its own at the compositor's rate. Each
/// frame takes the newest game frame the backend has finished
/// (`metal_d3d9_immersive_acquire`), never waiting for one, and draws it on
/// a 4:3 quad pinned 2.5 m ahead of where the head was when tracking first
/// reported it. The same image goes to every view: the quad sits at depth,
/// so the two eyes see it from their own positions as they would a real
/// screen. The game frame is held until this frame's command buffer
/// completes, so the game never writes into what the GPU still samples.
///
/// A stereo frame (two layers, `--stereo`) is drawn per eye instead: each
/// view gets its own eye's image on a plane centred on that eye's forward
/// axis, sized so the image subtends exactly the game's field of view
/// (`mtl_stereo_plane_mvp`), and seen through the view's own off-axis
/// projection. Each eye image is a symmetric frustum about its eye, and a
/// headset view is not: it reaches further out than towards the nose and
/// is mirrored between the eyes, so the same pixel is a different
/// direction in each. Stretched over the whole view, as stage A first did,
/// the game's centre landed some 40 degrees apart in the two eyes and
/// every eye saw two images. On the plane every pixel is in the direction
/// the game rendered it for, so points at infinity are parallel rays and
/// the world's disparity is the real one for the eyes' IPD. By default
/// the game keeps its own camera and the eye planes are pinned where the
/// first tracked head put them (`mtl_stereo_plane_mvp_fixed`), so the
/// stereo screen stands still in the room and the head looks across it.
/// The eye's offset the backend uses comes from the two views' own
/// positions; the HUD sits at the backend's HUD plane, which is also the
/// plane's distance, so the compositor's reprojection treats the picture
/// as something there rather than at infinity.
///
/// Full view (`--fullview`): a frame the game drew in play fills each
/// eye's whole view. The backend replays it into each eye's own off-axis
/// projection from that eye's own position, with the margin the late
/// rotation turns into, so here each slice goes on a plane covering
/// exactly its eye's tangents (`mtl_fullview_plane_mvp`), turned late, and
/// the rest of the view is black rather than the room. The views'
/// tangents and the eyes' positions go to the backend whenever they
/// change. Head look is on with it; a menu, dialogue or the sheet comes
/// as a stereo frame and goes on the head-locked plane as before.
///
/// Stage B, only when asked for (`RT_HEADLOOK=1` or `--headlook`): while
/// stereo frames are shown, each compositor frame hands the
/// head's yaw and pitch to the runtime (`rt_host_head_pose`), which turns
/// the game's view angles to them before the game's next frame. Stage C:
/// each frame then carries the pose it was drawn for, and the eye plane is
/// turned by the rotation from the head now to that pose
/// (`mtl_late_rotation`), so the world holds still between game frames
/// instead of dragging a frame behind the head. A frame with no pose (a
/// mono frame, the game's own camera before the head took it) keeps the
/// head-locked plane. The turn's size is logged every few seconds.
///
/// The class is `@unchecked Sendable` because its state is touched only by
/// the loop's thread after `start`, apart from the ARKit task, which only
/// runs the session.
final class ScreenCompositor: @unchecked Sendable {
    static let screenDistance: Float = 2.5
    static let screenWidth: Float = 2.5
    static let screenAspect: Float = 4.0 / 3.0

    private let renderer: LayerRenderer
    private let session = ARKitSession()
    private let worldTracking = WorldTrackingProvider()
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipeline: MTLRenderPipelineState
    private let stereoPipeline: MTLRenderPipelineState
    /// The tracked controllers' debug box, ray and tip, drawn over the
    /// frame with no depth test (`SpatialInput.rayShown`).
    private let handPipeline: MTLRenderPipelineState
    private let reticlePipeline: MTLRenderPipelineState
    /// The arm HUD's cards, premultiplied over the frame (`encodeArmHUD`).
    private let armPipeline: MTLRenderPipelineState
    /// The orbs, their pictures and labels printed on the glass (`encodeOrbs`).
    private let orbPipeline: MTLRenderPipelineState
    /// The same writing the depth of what each pixel shows, tested against
    /// our hands' (`RT_HANDS`) so a hand in front of an orb hides it and
    /// one behind is hidden; it writes none of its own.
    private let orbDepthPipeline: MTLRenderPipelineState
    private let orbDepthState: MTLDepthStencilState
    /// Our hands (`Hands/`, `encodeGloves`): the renderer, each hand's mesh
    /// and rig (0 left, 1 right), and each one's pose over time.
    private var gloveRenderer: HandRenderer?
    private var gloveMeshes: [HandMesh?] = [nil, nil]
    private var gloveRigs: [HandRig] = []
    private var glovePoses: [HandJointPose?] = [nil, nil]
    /// A model's own poses, per hand, when it came with them.
    private var glovePoseFiles: [HandPoseFile?] = [nil, nil]
    private var gloveBlenders = [HandPoseBlender(), HandPoseBlender()]
    private var gloveTime: Double = 0
    /// When a crosshair was last seen with a weapon out: a weapon with none
    /// for a while is held as a blade, not a gun.
    private var gloveCrosshairSeen: Double = 0
    private var gloveLogged = false
    /// The icon's scaling up, once per icon (`refreshIcon`): the game's
    /// texture to premultiplied pixels, Lanczos to four times the pixels,
    /// and MetalFX's colour joined to the Lanczos alpha where it is there.
    private let iconDecodePipeline: MTLRenderPipelineState
    private let iconLanczosPipeline: MTLRenderPipelineState
    private let iconMergePipeline: MTLRenderPipelineState
    /// A texture's alpha in a rectangle, to an 8 bit target read back on
    /// the CPU (`readAlpha`): the pictures' crops and the labels' lines.
    private let alphaMaskPipeline: MTLRenderPipelineState
    /// What the CPU found in them, filled in as each read completes.
    private let orbReads = OrbReads()
    private let overlayDepthState: MTLDepthStencilState
    /// The half IPD last given to the backend, in metres.
    private var halfIPD: Float = 0
    private let depthState: MTLDepthStencilState
    private let onEnd: @Sendable () -> Void
    /// The screen's model matrix, pinned at the first tracked head pose.
    private var model: [Float]?
    /// sRGB views of the backend's slot textures, so sampling decodes the
    /// game's gamma-encoded pixels and the sRGB target encodes them again.
    private var srgbViews: [ObjectIdentifier: MTLTexture] = [:]
    private var placeholder: MTLTexture
    private var framesDrawn: UInt64 = 0
    private var framesUntracked: UInt64 = 0
    private var checkedDevice = false
    /// The views' projections and transforms are logged once a space, and
    /// the stereo frames' field of view whenever it changes, so a device
    /// log says what every eye's frustum was. The backend fixes the plane
    /// for the session, so a second line means its back buffer changed
    /// shape, not that a camera zoomed.
    private var loggedViews = false
    private var loggedStereoTan: (Float, Float) = (0, 0)
    /// Whether a frame with the game on the screen has been submitted, so
    /// the window coordinator hears of the first one only.
    private var announcedGameFrame = false
    /// Whether the runtime was last told the head drives the camera, so
    /// the change is logged once.
    private var headDriving = false
    /// The late rotation's size over the current log period.
    private var late = LateRotationStats()
    /// Head look is asked for (`RT_HEADLOOK=1` or `--headlook`): the head
    /// drives the game's camera and the eye planes follow the head, turned
    /// late. Off by default: the game keeps its own camera and the stereo
    /// screen stands still in the room, as the mono one does.
    private let headLook = ProcessInfo.processInfo.environment["RT_HEADLOOK"] == "1"
        || CommandLine.arguments.contains("--headlook") || GameLauncher.wantsFullView
    /// The views last given to the backend for full view: both views'
    /// tangents then both eyes' positions, left first.
    private var fullViewViews: [Float] = []
    /// The fixed stereo screen's anchor, the first tracked head levelled
    /// (`mtl_stereo_anchor`), pinned with the mono screen's model.
    private var stereoAnchor: [Float]?
    /// Whether the last frame shown was full view, so the plane is placed
    /// again in front of the head when a menu or the sheet takes over.
    private var lastShownFullView = false
    /// The reticle's last logged decision and how many were logged
    /// (`noteReticle`).
    private var reticleLastKey = ""
    /// The reticle's ring radius in the shape's units, eased towards the
    /// game's spread (`reticleRing`).
    private var reticleRing: Float = ScreenCompositor.reticleRingRest
    /// The tightest the game's accuracy ring has been, as a half of its
    /// screen; 0.032 (a pistol standing still, build/hud-cap2) until one
    /// tighter is seen.
    private var tightestRingHalf: Float = 0.032
    /// The last ring seen, easing to the tightest while none is drawn.
    private var heldRingHalf: Float = 0.032
    private var reticleLogged = 0
    /// The orbs' pictures, 0 the weapon's and 1 the discipline's: the
    /// game's texture each last scaled up, held so its identity cannot be
    /// reused by another, and the result; and MetalFX's spatial scalers by
    /// input size, nil for a size it could not do.
    private var pictureSource: [MTLTexture?] = [nil, nil]
    private var picture: [MTLTexture?] = [nil, nil]
    private var pictureScalers: [SIMD2<Int>: MTLFXSpatialScaler?] = [:]
    /// The orbs between the hands (0) and their third person places (1).
    private var orbThirdPerson: Float = 0
    /// The body's facing for the third person places, the head's yaw
    /// smoothed, in radians; nil until first seen.
    private var orbBodyYaw: Float?
    private var loggedOrbSetup = false
    /// Each orb's liquid: its surface's tilt (x and z of its normal over
    /// y) and how fast that changes, sprung back to level, driven by the
    /// orb's acceleration from its last positions (`stepSlosh`).
    private var orbTilt: [SIMD2<Float>] = [.zero, .zero]
    private var orbTiltSpeed: [SIMD2<Float>] = [.zero, .zero]
    private var orbLastCentre: [SIMD3<Float>?] = [nil, nil]
    private var orbLastVelocity: [SIMD3<Float>] = [.zero, .zero]
    private var orbLastTime: Double = 0
    /// Where each piece the layout edit can pick up was drawn this frame
    /// (LayoutEditor), filled by the encoders and handed to the edit at
    /// the next frame's start; the head it was drawn for alongside.
    private var editPieces: [HUDPiece: LayoutEditor.Placed] = [:]
    private var editHead: simd_float4x4?
    private lazy var labelPainter = LabelPainter(device: device)

    /// Starts the loop for a layer the immersive space just made. `onEnd`
    /// runs on the loop's thread when the layer is invalidated.
    static func start(_ renderer: LayerRenderer, onEnd: @escaping @Sendable () -> Void) {
        do {
            let compositor = try ScreenCompositor(renderer: renderer, onEnd: onEnd)
            compositor.startTracking()
            let thread = Thread {
                compositor.run()
            }
            thread.name = "compositor"
            thread.qualityOfService = .userInteractive
            thread.start()
        } catch {
            log("could not set up: \(error)")
            onEnd()
        }
    }

    private init(renderer: LayerRenderer, onEnd: @escaping @Sendable () -> Void) throws {
        self.renderer = renderer
        self.onEnd = onEnd
        // The backend's device: the game's textures must be sampled on the
        // device that made them. The layer's is checked against it below.
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw CompositorError.noDevice
        }
        self.device = device
        self.queue = queue
        queue.label = "compositor"
        let configuration = renderer.configuration
        let library = try device.makeLibrary(source: Self.shaderSource, options: nil)
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.label = "virtual screen"
        descriptor.vertexFunction = library.makeFunction(name: "screen_vertex")
        descriptor.fragmentFunction = library.makeFunction(name: "screen_fragment")
        descriptor.colorAttachments[0].pixelFormat = configuration.colorFormat
        descriptor.depthAttachmentPixelFormat = configuration.depthFormat
        pipeline = try device.makeRenderPipelineState(descriptor: descriptor)
        descriptor.label = "stereo eye plane"
        descriptor.vertexFunction = library.makeFunction(name: "stereo_vertex")
        descriptor.fragmentFunction = library.makeFunction(name: "stereo_fragment")
        stereoPipeline = try device.makeRenderPipelineState(descriptor: descriptor)
        descriptor.label = "hand ray"
        descriptor.vertexFunction = library.makeFunction(name: "hand_vertex")
        descriptor.fragmentFunction = library.makeFunction(name: "hand_fragment")
        handPipeline = try device.makeRenderPipelineState(descriptor: descriptor)
        descriptor.label = "reticle"
        descriptor.vertexFunction = library.makeFunction(name: "reticle_vertex")
        descriptor.fragmentFunction = library.makeFunction(name: "reticle_fragment")
        let blend = descriptor.colorAttachments[0]!
        blend.isBlendingEnabled = true
        blend.sourceRGBBlendFactor = .one
        blend.sourceAlphaBlendFactor = .one
        blend.destinationRGBBlendFactor = .oneMinusSourceAlpha
        blend.destinationAlphaBlendFactor = .oneMinusSourceAlpha
        reticlePipeline = try device.makeRenderPipelineState(descriptor: descriptor)
        descriptor.label = "arm HUD"
        descriptor.vertexFunction = library.makeFunction(name: "arm_vertex")
        descriptor.fragmentFunction = library.makeFunction(name: "arm_fragment")
        armPipeline = try device.makeRenderPipelineState(descriptor: descriptor)
        descriptor.label = "orb"
        descriptor.vertexFunction = library.makeFunction(name: "orb_vertex")
        descriptor.fragmentFunction = library.makeFunction(name: "orb_fragment")
        orbPipeline = try device.makeRenderPipelineState(descriptor: descriptor)
        descriptor.label = "orb, depth tested"
        descriptor.fragmentFunction = library.makeFunction(name: "orb_fragment_depth")
        orbDepthPipeline = try device.makeRenderPipelineState(descriptor: descriptor)
        blend.isBlendingEnabled = false
        let offscreen = MTLRenderPipelineDescriptor()
        offscreen.vertexFunction = library.makeFunction(name: "icon_vertex")
        offscreen.colorAttachments[0].pixelFormat = .rgba8Unorm
        offscreen.label = "icon decode"
        offscreen.fragmentFunction = library.makeFunction(name: "icon_decode")
        iconDecodePipeline = try device.makeRenderPipelineState(descriptor: offscreen)
        offscreen.label = "icon Lanczos"
        offscreen.fragmentFunction = library.makeFunction(name: "icon_lanczos")
        iconLanczosPipeline = try device.makeRenderPipelineState(descriptor: offscreen)
        offscreen.label = "icon merge"
        offscreen.fragmentFunction = library.makeFunction(name: "icon_merge")
        iconMergePipeline = try device.makeRenderPipelineState(descriptor: offscreen)
        offscreen.label = "alpha mask"
        offscreen.fragmentFunction = library.makeFunction(name: "alpha_mask")
        offscreen.colorAttachments[0].pixelFormat = .r8Unorm
        alphaMaskPipeline = try device.makeRenderPipelineState(descriptor: offscreen)
        let depth = MTLDepthStencilDescriptor()
        // Reverse Z, as computeProjection gives: nearer is greater.
        depth.depthCompareFunction = .greater
        depth.isDepthWriteEnabled = true
        guard let depthState = device.makeDepthStencilState(descriptor: depth) else {
            throw CompositorError.noDevice
        }
        self.depthState = depthState
        // Over everything: the eye plane writes depth at the HUD distance and
        // would hide a ray that passes behind it.
        let overlay = MTLDepthStencilDescriptor()
        overlay.depthCompareFunction = .always
        overlay.isDepthWriteEnabled = false
        guard let overlayDepthState = device.makeDepthStencilState(descriptor: overlay) else {
            throw CompositorError.noDevice
        }
        self.overlayDepthState = overlayDepthState
        let tested = MTLDepthStencilDescriptor()
        tested.depthCompareFunction = .greaterEqual
        tested.isDepthWriteEnabled = false
        guard let orbDepthState = device.makeDepthStencilState(descriptor: tested) else {
            throw CompositorError.noDevice
        }
        self.orbDepthState = orbDepthState
        let texture = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm_srgb, width: 1, height: 1, mipmapped: false
        )
        guard let placeholder = device.makeTexture(descriptor: texture) else { throw CompositorError.noDevice }
        // Dark grey until the first game frame, so the screen's placement
        // shows at once.
        var grey: UInt32 = 0xFF18_1818
        placeholder.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: &grey, bytesPerRow: 4)
        self.placeholder = placeholder
        if SpatialInput.hands { setUpGloves(colourFormat: configuration.colorFormat, depthFormat: configuration.depthFormat) }
        Self.log(
            "layer layout \(Self.name(of: configuration.layout)), colour \(configuration.colorFormat.rawValue), "
                + "depth \(configuration.depthFormat.rawValue), \(renderer.properties.viewCount) views"
        )
    }

    private func startTracking() {
        let session = session
        let worldTracking = worldTracking
        Task.detached {
            guard WorldTrackingProvider.isSupported else {
                Self.log("world tracking is not supported here; the screen cannot be placed")
                return
            }
            do {
                try await session.run([worldTracking])
                Self.log("world tracking running")
            } catch {
                Self.log("world tracking did not start: \(error)")
            }
        }
        SpatialInput.shared.start()
    }

    private func run() {
        Self.log("compositor loop started")
        metal_d3d9_space_compositor_started()
        loop: while true {
            switch renderer.state {
            case .paused:
                renderer.waitUntilRunning()
            case .running:
                autoreleasepool {
                    renderFrame()
                }
            case .invalidated:
                break loop
            @unknown default:
                break loop
            }
        }
        session.stop()
        SpatialInput.shared.stop()
        // No cards without the compositor: the gauges go back to the panel.
        for side in 0..<2 { metal_d3d9_fullview_arm_hud(Int32(side), 0) }
        Self.log("compositor loop ended after \(framesDrawn) frames (\(framesUntracked) without a tracked head)")
        onEnd()
    }

    private func renderFrame() {
        guard let frame = renderer.queryNextFrame() else { return }
        frame.startUpdate()
        frame.endUpdate()
        guard let timing = frame.predictTiming() else { return }
        LayerRenderer.Clock().wait(until: timing.optimalInputTime)
        let drawables = frame.queryDrawables()
        guard !drawables.isEmpty else { return }
        frame.startSubmission()
        var shown = metal_d3d9_shown_frame()
        let haveFrame = metal_d3d9_immersive_acquire(&shown) != 0
        guard let commandBuffer = queue.makeCommandBuffer() else {
            if haveFrame { metal_d3d9_immersive_release(shown.slot) }
            frame.endSubmission()
            return
        }
        commandBuffer.label = "compositor frame"
        let source = haveFrame ? sourceTexture(for: shown) : nil
        let stereo = haveFrame && shown.layers == 2 && source?.textureType == .type2DArray
        let cursor = Self.guestCursor()
        if !loggedViews, let first = drawables.first {
            loggedViews = true
            logViews(of: first)
        }
        if metal_d3d9_fullview_enabled() != 0, let first = drawables.first {
            publishViews(of: first)
        }
        if stereo, loggedStereoTan != (shown.tan_x, shown.tan_y) {
            loggedStereoTan = (shown.tan_x, shown.tan_y)
            Self.log(
                "stereo frame: eye images span tan \(shown.tan_x) across and \(shown.tan_y) up and down "
                    + "(\(Self.degrees(shown.tan_x)) by \(Self.degrees(shown.tan_y)) degrees), half IPD "
                    + "\(shown.half_ipd_m * 1000) mm, HUD plane \(shown.hud_m) m"
            )
        }
        var publishedHead = false
        // The controllers once a frame, on this thread: the anchors are not
        // safe to read from two.
        let trackableTime = LayerRenderer.Clock.Instant.epoch.duration(to: drawables[0].frameTiming.trackableAnchorTime)
        let hands = SpatialInput.shared.frame(at: trackableTime.seconds)
        // Which cards can be shown, for the game's next frames: a side
        // whose controller has no position keeps its gauge on the panel.
        for hand in hands {
            let tracked = hand.valid && hand.state >= 2
            let mode: Int32 = !tracked ? 0 : SpatialInput.armHUD ? 1 : SpatialInput.orbHUD ? 2 : 0
            metal_d3d9_fullview_arm_hud(Int32(hand.side.rawValue), mode)
        }
        if stereo, shown.fullview != 0, SpatialInput.orbHUD {
            refreshPicture(shown.weapon_texture, side: 0, into: commandBuffer)
            refreshPicture(shown.icon_texture, side: 1, into: commandBuffer)
            readLabels(shown: shown, into: commandBuffer)
        }
        // The layout edit, from where the pieces were drawn last frame: its
        // chord is watched always, its grabs only act while it is open.
        let editControllers = (0..<2).map { i -> LayoutEditor.Controller? in
            guard i < hands.count, hands[i].valid, hands[i].state >= 2 else { return nil }
            return LayoutEditor.Controller(grip: hands[i].originFromGrip, aim: hands[i].originFromAim)
        }
        LayoutEditor.shared.step(pieces: editPieces, controllers: editControllers, head: editHead ?? matrix_identity_float4x4)
        editPieces = [:]
        for drawable in drawables {
            let time = LayerRenderer.Clock.Instant.epoch.duration(to: drawable.frameTiming.presentationTime)
            let anchor = worldTracking.queryDeviceAnchor(atTimestamp: time.seconds)
            drawable.deviceAnchor = anchor
            if let anchor, anchor.isTracked { editHead = anchor.originFromAnchorTransform }
            if let anchor, anchor.isTracked, model == nil {
                model = Self.placeScreen(head: anchor.originFromAnchorTransform)
                stereoAnchor = Self.stereoAnchor(head: anchor.originFromAnchorTransform)
                Self.log("head look: " + (headLook ? "asked for, the head drives the view" : "off, the screen stands still"))
            }
            if headLook, !publishedHead {
                publishedHead = true
                publishHead(anchor: anchor, stereo: stereo)
            }
            // A menu, the sheet or the inventory opening out of full view
            // comes up in front of the head, not where the screen was
            // pinned when the game loaded, and then stays put while open.
            let fullViewFrame = stereo && shown.fullview != 0 && source != nil
            if GameLauncher.wantsFullView, stereo, !fullViewFrame, lastShownFullView,
               let anchor, anchor.isTracked {
                stereoAnchor = Self.stereoAnchor(head: anchor.originFromAnchorTransform)
                Self.log("theater screen placed again in front of the head")
            }
            if source != nil { lastShownFullView = fullViewFrame }
            if stereo, shown.fullview != 0, let source {
                encodeFullView(
                    drawable: drawable, anchor: anchor, source: source, shown: shown, into: commandBuffer
                )
            } else if stereo, let source {
                encodeStereo(
                    drawable: drawable, anchor: anchor, source: source, shown: shown, cursor: cursor, into: commandBuffer
                )
            } else {
                encode(drawable: drawable, anchor: anchor, source: source ?? placeholder, cursor: cursor, into: commandBuffer)
            }
            if SpatialInput.hands, gloveRenderer != nil {
                encodeGloves(
                    drawable: drawable, anchor: anchor, hands: hands, shown: shown,
                    fullView: stereo && shown.fullview != 0 && source != nil, menu: SpatialInput.rayShown || cursor.z > 0,
                    into: commandBuffer
                )
            }
            if stereo, shown.fullview != 0, source != nil, SpatialInput.armHUD {
                encodeArmHUD(drawable: drawable, anchor: anchor, hands: hands, shown: shown, into: commandBuffer)
            }
            if stereo, shown.fullview != 0, source != nil, SpatialInput.orbHUD {
                encodeOrbs(drawable: drawable, anchor: anchor, hands: hands, shown: shown, into: commandBuffer)
            }
            // The rays only while a menu's cursor is up (the inventory, the
            // sheet), where they point; in play they would be clutter.
            if SpatialInput.rayShown || cursor.z > 0 {
                encodeHands(drawable: drawable, anchor: anchor, hands: hands, into: commandBuffer)
            } else if SpatialInput.handAim, stereo, shown.fullview != 0 {
                encodeReticle(drawable: drawable, anchor: anchor, hands: hands, shown: shown, into: commandBuffer)
            }
            if LayoutEditor.shared.isActive {
                encodeEditOverlay(drawable: drawable, anchor: anchor, into: commandBuffer)
            }
            drawable.encodePresent(commandBuffer: commandBuffer)
        }
        if GameLauncher.wantsFullView, stereo, cursor.z > 0, shown.fullview == 0, let plane = stereoAnchor {
            pointCursor(
                hands: hands, anchor: plane, tanX: shown.tan_x, tanY: shown.tan_y,
                distance: shown.hud_m > 0 ? shown.hud_m : METAL_D3D9_STEREO_HUD_M
            )
        }
        late.logIfDue()
        if haveFrame {
            let slot = shown.slot
            commandBuffer.addCompletedHandler { _ in
                metal_d3d9_immersive_release(slot)
            }
        }
        // The first game frame drawn on the placed screen: once the GPU has
        // finished it, the app's window can go.
        if haveFrame, model != nil, !announcedGameFrame {
            announcedGameFrame = true
            commandBuffer.addCompletedHandler { _ in
                // Recorded here, on the completion's thread, so the space's
                // order holds it even if the hop below runs late.
                metal_d3d9_space_first_frame_shown()
                Task { @MainActor in
                    WindowCoordinator.shared.screenShowedFirstFrame()
                }
            }
        }
        commandBuffer.commit()
        frame.endSubmission()
        framesDrawn += 1
    }

    /// The head's yaw, pitch and position at this frame's presentation, for
    /// the game's camera, while stereo frames are shown and the head is tracked. Off
    /// otherwise: the virtual screen is world locked and the game keeps its
    /// own camera there, and an untracked head has no pose to give. The
    /// runtime keeps the view where it is across each switch, so neither
    /// way makes the view jump.
    private func publishHead(anchor: DeviceAnchor?, stereo: Bool) {
        guard stereo, let anchor, anchor.isTracked else {
            if headDriving {
                headDriving = false
                rt_host_head_pose(0, 0, 0)
                Self.log("head look: off (" + (stereo ? "head not tracked" : "no stereo frame") + ")")
            }
            return
        }
        var m = anchor.originFromAnchorTransform
        var yaw: Float = 0
        var pitch: Float = 0
        withUnsafeBytes(of: &m) { bytes in
            mtl_head_yaw_pitch(bytes.bindMemory(to: Float.self).baseAddress, &yaw, &pitch)
        }
        rt_host_head_pose(1, yaw, pitch)
        // Where the head is as well, for hand aim to draw the gun where the
        // hand is from it.
        let position = m.columns.3
        rt_host_head_position(position.x, position.y, position.z)
        if !headDriving {
            headDriving = true
            Self.log("head look: on, head at yaw \(yaw) pitch \(pitch)")
        }
    }

    /// One pass per view, into the texture and slice its map names, with
    /// its viewport: right for dedicated, layered and shared layouts alike.
    /// A target shared by two views is cleared only by the first.
    private func encode(
        drawable: LayerRenderer.Drawable,
        anchor: DeviceAnchor?,
        source: MTLTexture,
        cursor: SIMD4<Float>,
        into commandBuffer: MTLCommandBuffer
    ) {
        let tracked = anchor?.isTracked ?? false
        if !tracked { framesUntracked += 1 }
        var cleared = Set<[Int]>()
        for (index, view) in drawable.views.enumerated() {
            let map = view.textureMap
            let key = [map.textureIndex, map.sliceIndex]
            let first = cleared.insert(key).inserted
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = drawable.colorTextures[map.textureIndex]
            pass.colorAttachments[0].slice = map.sliceIndex
            pass.colorAttachments[0].loadAction = first ? .clear : .load
            pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
            pass.colorAttachments[0].storeAction = .store
            pass.depthAttachment.texture = drawable.depthTextures[map.textureIndex]
            pass.depthAttachment.slice = map.sliceIndex
            pass.depthAttachment.loadAction = first ? .clear : .load
            pass.depthAttachment.clearDepth = 0
            pass.depthAttachment.storeAction = .store
            guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else { continue }
            encoder.label = "view \(index)"
            encoder.setViewport(map.viewport)
            // Without a tracked head there is no pose to place the screen
            // in, so the view stays clear rather than drawing it head-locked.
            if tracked, let anchor, let model {
                let projection = drawable.computeProjection(convention: .rightUpBack, viewIndex: index)
                let originFromView = anchor.originFromAnchorTransform * view.transform
                var uniforms = ScreenUniforms(
                    mvp: Self.mvp(projection: projection, originFromView: originFromView, model: model),
                    cursor: cursor
                )
                encoder.setRenderPipelineState(pipeline)
                encoder.setDepthStencilState(depthState)
                encoder.setCullMode(.none)
                encoder.setVertexBytes(&uniforms, length: MemoryLayout<ScreenUniforms>.stride, index: 0)
                encoder.setFragmentBytes(&uniforms, length: MemoryLayout<ScreenUniforms>.stride, index: 0)
                encoder.setFragmentTexture(source, index: 0)
                encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            }
            encoder.endEncoding()
        }
    }

    /// Stereo: each view draws its own eye's slice on that eye's plane
    /// (`mtl_stereo_plane_mvp`), through its own projection. The left view
    /// is the one further to the left in the device's space, whatever order
    /// the views come in, and the distance between the two is the IPD the
    /// backend offsets the eyes by. The plane is turned by the late
    /// rotation when the frame carries the head pose it was drawn for and
    /// the head is tracked, and is head locked otherwise, which needs no
    /// tracked head; the anchor is still given to the drawable for the
    /// compositor's reprojection. Outside the plane nothing is drawn and no
    /// depth written, so the room shows there, as around the screen, and
    /// shows at the edge the turn uncovers too: the frame cannot fill what
    /// it was not drawn for.
    private func encodeStereo(
        drawable: LayerRenderer.Drawable,
        anchor: DeviceAnchor?,
        source: MTLTexture,
        shown: metal_d3d9_shown_frame,
        cursor: SIMD4<Float>,
        into commandBuffer: MTLCommandBuffer
    ) {
        if !(anchor?.isTracked ?? false) { framesUntracked += 1 }
        let xs = drawable.views.map { $0.transform.columns.3.x }
        if xs.count == 2 {
            let half = abs(xs[1] - xs[0]) / 2
            if half > 0.02, half < 0.05, abs(half - halfIPD) > 0.0005 {
                halfIPD = half
                metal_d3d9_stereo_set_half_ipd(half)
            }
        }
        let leftmost = xs.min() ?? 0
        let distance = shown.hud_m > 0 ? shown.hud_m : METAL_D3D9_STEREO_HUD_M
        // The late rotation, one for both eyes: the head now against the
        // pose the frame was drawn for. Without both, the plane stays head
        // locked, as before the camera followed the head.
        var turn = matrix_identity_float4x4
        // Under full view this plane only shows frames the head does not
        // drive (menus, the sheet, the inventory, head look off by hand), so
        // it stands still in the room, placed again in front of the head
        // each time one opens.
        let fixedAnchor = headLook && !GameLauncher.wantsFullView ? nil : stereoAnchor
        let fixedHead = (anchor?.isTracked ?? false) ? anchor?.originFromAnchorTransform : nil
        if fixedAnchor != nil {
            // Nothing to turn late: the camera is the game's own.
        } else if shown.pose_valid != 0, let anchor, anchor.isTracked {
            turn = Self.lateRotation(
                originFromDevice: anchor.originFromAnchorTransform, yaw: shown.pose_yaw_deg, pitch: shown.pose_pitch_deg
            )
            late.add(degrees: Self.rotationDegrees(turn), age: shown.age_ms)
        } else {
            late.addUnturned()
        }
        var cleared = Set<[Int]>()
        for (index, view) in drawable.views.enumerated() {
            let map = view.textureMap
            let key = [map.textureIndex, map.sliceIndex]
            let first = cleared.insert(key).inserted
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = drawable.colorTextures[map.textureIndex]
            pass.colorAttachments[0].slice = map.sliceIndex
            pass.colorAttachments[0].loadAction = first ? .clear : .load
            pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
            pass.colorAttachments[0].storeAction = .store
            pass.depthAttachment.texture = drawable.depthTextures[map.textureIndex]
            pass.depthAttachment.slice = map.sliceIndex
            pass.depthAttachment.loadAction = first ? .clear : .load
            pass.depthAttachment.clearDepth = 0
            pass.depthAttachment.storeAction = .store
            guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else { continue }
            encoder.label = "stereo view \(index)"
            encoder.setViewport(map.viewport)
            let right = xs.count == 2 && view.transform.columns.3.x > leftmost
            let projection = drawable.computeProjection(convention: .rightUpBack, viewIndex: index)
            // The cursor is the guest's, drawn here rather than by the game,
            // so it is moved to the HUD plane as the backend moves the HUD:
            // by e / d in direction, which is that over 2 tan_x in uv.
            var eyeCursor = cursor
            if shown.tan_x > 0, distance > 0 {
                let e = right ? shown.half_ipd_m : -shown.half_ipd_m
                eyeCursor.x -= e / distance / (2 * shown.tan_x)
            }
            // An untracked head has no pose to place the fixed screen from,
            // so that frame draws nothing, as the mono screen does.
            let mvp: simd_float4x4
            if let fixedAnchor {
                guard let fixedHead else {
                    encoder.endEncoding()
                    continue
                }
                mvp = Self.fixedPlaneMVP(
                    projection: projection, deviceFromView: view.transform, originFromDevice: fixedHead,
                    anchor: fixedAnchor, tanX: shown.tan_x, tanY: shown.tan_y, distance: distance
                )
            } else {
                mvp = Self.planeMVP(
                    projection: projection, deviceFromView: view.transform, turn: turn, tanX: shown.tan_x,
                    tanY: shown.tan_y, distance: distance
                )
            }
            encoder.setRenderPipelineState(stereoPipeline)
            encoder.setDepthStencilState(depthState)
            encoder.setCullMode(.none)
            var uniforms = StereoUniforms(
                mvp: mvp,
                cursor: eyeCursor,
                slice: right ? 1 : 0
            )
            encoder.setVertexBytes(&uniforms, length: MemoryLayout<StereoUniforms>.stride, index: 0)
            encoder.setFragmentBytes(&uniforms, length: MemoryLayout<StereoUniforms>.stride, index: 0)
            encoder.setFragmentTexture(source, index: 0)
            encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            encoder.endEncoding()
        }
    }

    /// Full view: each view draws its own eye's slice over its whole view,
    /// on a plane covering exactly the tangents that slice was drawn for
    /// (`mtl_fullview_plane_mvp`), turned by the late rotation when the
    /// frame carries its pose and the head is tracked. The view is cleared
    /// to opaque black first, so an edge a fast turn uncovers past the
    /// margin is black, not the room. The guest's cursor is not drawn: in
    /// play the game hides it, and a menu comes on the plane.
    /// How far in from the eye image's border full view fades to black, in
    /// uv. In first person the border sits at the edge of the display, so
    /// this is unseen; with the world held still (dialogue, third person,
    /// a menu) a turned head finds it, and a hard edge there read as a big
    /// window rather than the world (headset run, 2026-11-21).
    /// `RT_FULLVIEW_EDGE` sets it, 0 for none.
    private static let fullViewEdge: Float = {
        if let v = ProcessInfo.processInfo.environment["RT_FULLVIEW_EDGE"], let f = Float(v), f >= 0, f <= 0.5 {
            return f
        }
        return 0.12
    }()

    private func encodeFullView(
        drawable: LayerRenderer.Drawable,
        anchor: DeviceAnchor?,
        source: MTLTexture,
        shown: metal_d3d9_shown_frame,
        dim: Float = 1,
        into commandBuffer: MTLCommandBuffer
    ) {
        if !(anchor?.isTracked ?? false) { framesUntracked += 1 }
        let xs = drawable.views.map { $0.transform.columns.3.x }
        let leftmost = xs.min() ?? 0
        let distance = shown.hud_m > 0 ? shown.hud_m : METAL_D3D9_STEREO_HUD_M
        var turn = matrix_identity_float4x4
        var turned: Float = 0
        if shown.pose_valid != 0, let anchor, anchor.isTracked {
            turn = Self.lateRotation(
                originFromDevice: anchor.originFromAnchorTransform, yaw: shown.pose_yaw_deg, pitch: shown.pose_pitch_deg
            )
            turned = Self.rotationDegrees(turn)
            late.add(degrees: turned, age: shown.age_ms)
        } else {
            late.addUnturned()
        }
        // The border fades in only as the frame is turned away from the
        // pose it was drawn for: a few degrees of lag in first person
        // leave it off, a held view with the head turned brings it in.
        let edge = Self.fullViewEdge * min(1, max(0, (turned - 3) / 7))
        var eyeTan = shown.eye_tan
        let tangents = withUnsafeBytes(of: &eyeTan) { Array($0.bindMemory(to: Float.self)) }
        var cleared = Set<[Int]>()
        for (index, view) in drawable.views.enumerated() {
            let map = view.textureMap
            let key = [map.textureIndex, map.sliceIndex]
            let first = cleared.insert(key).inserted
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = drawable.colorTextures[map.textureIndex]
            pass.colorAttachments[0].slice = map.sliceIndex
            pass.colorAttachments[0].loadAction = first ? .clear : .load
            pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
            pass.colorAttachments[0].storeAction = .store
            pass.depthAttachment.texture = drawable.depthTextures[map.textureIndex]
            pass.depthAttachment.slice = map.sliceIndex
            pass.depthAttachment.loadAction = first ? .clear : .load
            pass.depthAttachment.clearDepth = 0
            pass.depthAttachment.storeAction = .store
            guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else { continue }
            encoder.label = "full view \(index)"
            encoder.setViewport(map.viewport)
            let right = xs.count == 2 && view.transform.columns.3.x > leftmost
            let projection = drawable.computeProjection(convention: .rightUpBack, viewIndex: index)
            let eye = Array(tangents[(right ? 4 : 0)..<(right ? 8 : 4)])
            var uniforms = StereoUniforms(
                mvp: Self.fullViewMVP(
                    projection: projection, deviceFromView: view.transform, turn: turn, tangents: eye, distance: distance
                ),
                cursor: .zero,
                slice: right ? 1 : 0,
                edge: edge,
                dim: dim
            )
            encoder.setRenderPipelineState(stereoPipeline)
            encoder.setDepthStencilState(depthState)
            encoder.setCullMode(.none)
            encoder.setVertexBytes(&uniforms, length: MemoryLayout<StereoUniforms>.stride, index: 0)
            encoder.setFragmentBytes(&uniforms, length: MemoryLayout<StereoUniforms>.stride, index: 0)
            encoder.setFragmentTexture(source, index: 0)
            encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            encoder.endEncoding()
        }
    }

    /// The debug box at each tracked controller's grip, a 2 m ray from its
    /// aim and a dot at the ray's end, left blue and right red, the aiming
    /// hand's ray brighter. Drawn after the frame in each view with no depth
    /// test, in theater and full view alike, so it shows where ARKit puts
    /// the controller against where it is. Needs a tracked head to draw in
    /// the room's space.
    private func encodeHands(
        drawable: LayerRenderer.Drawable,
        anchor: DeviceAnchor?,
        hands: [HandPose],
        into commandBuffer: MTLCommandBuffer
    ) {
        guard let anchor, anchor.isTracked, hands.contains(where: \.valid) else { return }
        let aimSide = SpatialInput.aimSide
        for (index, view) in drawable.views.enumerated() {
            let map = view.textureMap
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = drawable.colorTextures[map.textureIndex]
            pass.colorAttachments[0].slice = map.sliceIndex
            pass.colorAttachments[0].loadAction = .load
            pass.colorAttachments[0].storeAction = .store
            pass.depthAttachment.texture = drawable.depthTextures[map.textureIndex]
            pass.depthAttachment.slice = map.sliceIndex
            pass.depthAttachment.loadAction = .load
            pass.depthAttachment.storeAction = .store
            guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else { continue }
            encoder.label = "hands \(index)"
            encoder.setViewport(map.viewport)
            encoder.setRenderPipelineState(handPipeline)
            encoder.setDepthStencilState(overlayDepthState)
            encoder.setCullMode(.none)
            let projection = drawable.computeProjection(convention: .rightUpBack, viewIndex: index)
            let viewFromOrigin = (anchor.originFromAnchorTransform * view.transform).inverse
            let viewProjection = projection * viewFromOrigin
            for hand in hands where hand.valid {
                let aims = hand.side == aimSide
                let base: SIMD4<Float> = hand.side == .left ? [0.25, 0.55, 1.0, 1] : [1.0, 0.25, 0.2, 1]
                let ray = aims ? base : base * SIMD4<Float>(0.6, 0.6, 0.6, 1)
                // Orientation only: no position to trust, so a dim box only.
                let positioned = hand.state >= 2
                var parts: [(simd_float4x4, SIMD4<Float>)] = [
                    (hand.originFromGrip * Self.scale(0.03, 0.03, 0.05), positioned ? base : base * 0.4)
                ]
                if positioned {
                    parts.append((hand.originFromAim * Self.translate(0, 0, -1) * Self.scale(0.004, 0.004, 2), ray))
                    parts.append((hand.originFromAim * Self.translate(0, 0, -2) * Self.scale(0.02, 0.02, 0.02), ray))
                }
                for (model, colour) in parts {
                    var uniforms = HandUniforms(mvp: viewProjection * model, colour: colour)
                    encoder.setVertexBytes(&uniforms, length: MemoryLayout<HandUniforms>.stride, index: 0)
                    encoder.setFragmentBytes(&uniforms, length: MemoryLayout<HandUniforms>.stride, index: 0)
                    encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 36)
                }
            }
            encoder.endEncoding()
        }
    }

    /// Where the gun hand's shots go, in play: in first person a mark 8 m
    /// out along its aim turned down by the hand aim tilt, as the runtime
    /// turns the shot, so the reticle is at the hand rather than at the
    /// middle of the head's view. In third person at the point the
    /// backend finds the shot's own ray lands on (aim_shown), whatever
    /// the weapon; the game's own mark is a world draw and stays drawn
    /// beside it. Drawn over the frame with no depth test, like the rays.
    private func encodeReticle(
        drawable: LayerRenderer.Drawable,
        anchor: DeviceAnchor?,
        hands: [HandPose],
        shown: metal_d3d9_shown_frame,
        into commandBuffer: MTLCommandBuffer
    ) {
        let thirdPerson = rt_host_third_person() != 0
        guard let anchor, anchor.isTracked else {
            noteReticle("none (head not tracked)", thirdPerson: thirdPerson, shown: shown)
            return
        }
        let head = anchor.originFromAnchorTransform.columns.3
        // The game's camera sits between the eyes, so the shot starts there:
        // both branches put the mark on a ray from the eyes' midpoint.
        let eyes = drawable.views.reduce(SIMD4<Float>(0, 0, 0, 0)) { $0 + $1.transform.columns.3 }
            / Float(max(drawable.views.count, 1))
        let mid = anchor.originFromAnchorTransform * SIMD4(eyes.x, eyes.y, eyes.z, 1)
        let model: simd_float4x4
        if thirdPerson {
            // In third person the shot leaves the player's eye along the
            // game's view angles, not the camera's middle, and the game's
            // own mark is drawn into the world where it lands, with a shape
            // and spread that change per weapon. So the backend takes the
            // landing point itself: the engine's own trace of the shot's ray
            // when the runtime made one, else that ray against the frame's
            // depth (aim_shown), and the mark goes on that point: at its
            // tangents off the drawn forward and its depth, seen from the
            // game's camera, which is the eyes' midpoint, so both eyes see
            // it with the world's parallax. No point (a menu, a load, the
            // ray behind the camera) draws none.
            guard shown.pose_valid != 0, shown.aim_shown != 0, shown.aim_m > 0 else {
                noteReticle("none (no aim point in third person)", thirdPerson: thirdPerson, shown: shown)
                return
            }
            let drawn = Self.viewRotation(yaw: shown.pose_yaw_deg, pitch: shown.pose_pitch_deg)
            let depth = shown.aim_m
            let along = SIMD4<Float>(shown.aim_tan.0 * depth, shown.aim_tan.1 * depth, -depth, 0)
            let at = mid + drawn * along
            // A fixed angle, the hand reticle's; its ring is at rest here,
            // since the game draws no accuracy ring in third person.
            let size = 0.022 * depth * LayoutStore.shared.current.reticleScale
            // Facing along the drawn forward and level with the world, as
            // the frame is.
            model = simd_float4x4(drawn.columns.0 * size, drawn.columns.1 * size, drawn.columns.2, SIMD4(at.x, at.y, at.z, 1))
            noteReticle(shown.aim_hit != 0 ? "third person aim point" : "third person aim, far end (no surface)",
                        thirdPerson: thirdPerson, shown: shown)
        } else {
            // Out along the aim and sized from the layout (the edit's).
            let reticle = LayoutStore.shared.current
            let distance = reticle.reticleDistance
            let size = distance * 0.022 * reticle.reticleScale
            let hand = hands[SpatialInput.aimSide.rawValue]
            guard hand.valid, hand.state >= 2 else {
                noteReticle("none (aiming hand not tracked)", thirdPerson: thirdPerson, shown: shown)
                return
            }
            let tilt = -SpatialInput.aimTilt * .pi / 180
            let (c, sn) = (cos(tilt), sin(tilt))
            var turn = matrix_identity_float4x4
            turn.columns.1 = [0, c, sn, 0]
            turn.columns.2 = [0, -sn, c, 0]
            // The shot leaves the game's camera, between the eyes, along the
            // hand's aim turned down: so the mark is on that ray, not on one
            // from the hand, which ran parallel a forearm away and read as
            // slightly off at every range (a headset run, 2026-11-23).
            let aim = hand.originFromAim * turn
            let forward = -SIMD3(aim.columns.2.x, aim.columns.2.y, aim.columns.2.z)
            let centre = SIMD3(mid.x, mid.y, mid.z) + simd_normalize(forward) * distance
            // The square turned to face the head, so the shape reads the same
            // however the hand rolls.
            let back = simd_normalize(SIMD3(head.x, head.y, head.z) - centre)
            var right = simd_cross(SIMD3<Float>(0, 1, 0), back)
            right = simd_length(right) > 1e-4 ? simd_normalize(right) : SIMD3(1, 0, 0)
            let up = simd_cross(back, right)
            model = simd_float4x4(
                SIMD4(right * size, 0), SIMD4(up * size, 0), SIMD4(back, 0), SIMD4(centre, 1)
            )
            editPieces[.reticle] = LayoutEditor.Placed(
                world: simd_float4x4(SIMD4(right, 0), SIMD4(up, 0), SIMD4(back, 0), SIMD4(centre, 1)),
                radius: max(size * 0.6, 0.1), anchor: HUDLayoutMath.Frame(aim)
            )
            noteReticle("hand", thirdPerson: thirdPerson, shown: shown)
        }
        let gauges = Self.reticleGauges()
        let ring = reticleRing(shown: shown)
        let gaugeRadius = max(Self.reticleGaugeRest, ring * 1.6)
        for (index, view) in drawable.views.enumerated() {
            let map = view.textureMap
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = drawable.colorTextures[map.textureIndex]
            pass.colorAttachments[0].slice = map.sliceIndex
            pass.colorAttachments[0].loadAction = .load
            pass.colorAttachments[0].storeAction = .store
            pass.depthAttachment.texture = drawable.depthTextures[map.textureIndex]
            pass.depthAttachment.slice = map.sliceIndex
            pass.depthAttachment.loadAction = .load
            pass.depthAttachment.storeAction = .store
            guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else { continue }
            encoder.label = "reticle \(index)"
            encoder.setViewport(map.viewport)
            encoder.setRenderPipelineState(reticlePipeline)
            encoder.setDepthStencilState(overlayDepthState)
            encoder.setCullMode(.none)
            let projection = drawable.computeProjection(convention: .rightUpBack, viewIndex: index)
            let viewFromOrigin = (anchor.originFromAnchorTransform * view.transform).inverse
            let withGauges = gauges.health >= 0 || gauges.blood >= 0
            var uniforms = ReticleUniforms(
                mvp: projection * viewFromOrigin * model, colour: SpatialInput.reticleColour,
                style: SpatialInput.reticleStyle, health: gauges.health, blood: gauges.blood,
                extent: max(withGauges ? 1.25 : 1, (withGauges ? gaugeRadius : ring * 1.36) + 0.13),
                ring: ring, gaugeRadius: gaugeRadius
            )
            encoder.setVertexBytes(&uniforms, length: MemoryLayout<ReticleUniforms>.stride, index: 0)
            encoder.setFragmentBytes(&uniforms, length: MemoryLayout<ReticleUniforms>.stride, index: 0)
            encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            encoder.endEncoding()
        }
    }

    /// The arm HUD: each side's gauge, cut out of the frame by the backend
    /// at the back buffer's own pixels, on a card 14 cm tall just above its
    /// controller's grip and a little towards the body, leaning back so its
    /// face turns up to the eyes (`SpatialInput.armHUDOffset`,
    /// `armHUDTilt`). At the hand's predicted pose, so it holds to the
    /// controller at display rate whatever the game's frame rate. Drawn
    /// over the frame with no depth test, like the reticle.
    /// The layout edit's overlay: a ring on each piece that can be picked
    /// up, the held piece's values on a panel above it, and the controls
    /// on a strip ahead and a little low. Painted textures through the arm
    /// cards' pipeline, over everything, facing the head.
    private func encodeEditOverlay(drawable: LayerRenderer.Drawable, anchor: DeviceAnchor?, into commandBuffer: MTLCommandBuffer) {
        guard let anchor, anchor.isTracked else { return }
        let overlay = LayoutEditor.shared.overlay
        let head = anchor.originFromAnchorTransform
        let headPosition = SIMD3<Float>(head.columns.3.x, head.columns.3.y, head.columns.3.z)
        let metresPerPixel: Float = 0.0005
        func facingHead(_ centre: SIMD3<Float>, width: Float, height: Float) -> simd_float4x4 {
            let back = simd_length(headPosition - centre) > 1e-4 ? simd_normalize(headPosition - centre) : SIMD3(0, 0, 1)
            var right = simd_cross(SIMD3<Float>(0, 1, 0), back)
            right = simd_length(right) > 1e-4 ? simd_normalize(right) : SIMD3(1, 0, 0)
            let up = simd_cross(back, right)
            return simd_float4x4(SIMD4(right * width, 0), SIMD4(up * height, 0), SIMD4(back, 0), SIMD4(centre, 1))
        }
        var quads: [(MTLTexture, simd_float4x4)] = []
        if let ring = labelPainter.ring {
            for mark in overlay.marks {
                let d = mark.radius * (mark.held ? 2.5 : 2.2)
                quads.append((ring, facingHead(mark.centre, width: d, height: d)))
            }
        }
        for mark in overlay.marks {
            guard let text = mark.text, let texture = labelPainter.texture(for: text) else { continue }
            let w = Float(texture.width) * metresPerPixel, h = Float(texture.height) * metresPerPixel
            let up = SIMD3<Float>(0, 1, 0)
            quads.append((texture, facingHead(mark.centre + up * (mark.radius * 1.3 + h / 2), width: w, height: h)))
        }
        // The gizmo on the selected piece: red, green and blue for its
        // right, up and forward, gold where a hand is over a handle or
        // dragging it.
        if let mark = overlay.gizmo {
            let g = mark.gizmo
            let colours: [SIMD3<Float>] = [[0.92, 0.22, 0.2], [0.35, 0.85, 0.3], [0.3, 0.5, 0.98]]
            let lit = SIMD3<Float>(1, 0.84, 0.3)
            func colour(_ h: GizmoHandle, _ base: SIMD3<Float>) -> SIMD3<Float> {
                mark.active == h || mark.hovered.contains(h) ? lit : base
            }
            for h in g.handles {
                switch h {
                case .axis(let i):
                    let a = g.axes[i]
                    guard let texture = labelPainter.handle(.arrow, colour: colour(h, colours[i])) else { continue }
                    let mid = g.centre + a * (g.arrowStart + g.arrowLength / 2)
                    // Lengthwise along the axis, its face turned to the head.
                    let toHead = simd_normalize(headPosition - mid)
                    var across = simd_cross(toHead, a)
                    if simd_length(across) < 1e-4 { across = simd_cross(SIMD3(0, 1, 0), a) }
                    across = simd_normalize(across)
                    let width = g.arrowLength * (mark.active == h ? 0.32 : 0.25)
                    quads.append((texture, simd_float4x4(SIMD4(a * g.arrowLength, 0), SIMD4(across * width, 0),
                                                         SIMD4(simd_cross(a, across), 0), SIMD4(mid, 1))))
                case .ring(let i):
                    let a = g.axes[i]
                    guard let texture = labelPainter.handle(.ring, colour: colour(h, colours[i])) else { continue }
                    let u = g.axes[(i + 1) % 3], v = g.axes[(i + 2) % 3]
                    let d = g.ringRadius * 2
                    quads.append((texture, simd_float4x4(SIMD4(u * d, 0), SIMD4(v * d, 0), SIMD4(a, 0), SIMD4(g.centre, 1))))
                case .scale:
                    guard let texture = labelPainter.handle(.square, colour: colour(h, [0.85, 0.85, 0.85])) else { continue }
                    quads.append((texture, facingHead(g.scalePoint, width: g.squareSize, height: g.squareSize)))
                }
            }
        }
        if let banner = overlay.banner, let texture = labelPainter.texture(for: banner) {
            let forward = -simd_normalize(SIMD3<Float>(head.columns.2.x, 0, head.columns.2.z))
            // Ahead and a little low, inside the view's lower edge, so it
            // is seen without looking down.
            let centre = headPosition + forward * 1.2 + SIMD3(0, -0.22, 0)
            let scale = metresPerPixel * 1.0
            quads.append((texture, facingHead(centre, width: Float(texture.width) * scale, height: Float(texture.height) * scale)))
        }
        guard !quads.isEmpty else { return }
        for (index, view) in drawable.views.enumerated() {
            let map = view.textureMap
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = drawable.colorTextures[map.textureIndex]
            pass.colorAttachments[0].slice = map.sliceIndex
            pass.colorAttachments[0].loadAction = .load
            pass.colorAttachments[0].storeAction = .store
            pass.depthAttachment.texture = drawable.depthTextures[map.textureIndex]
            pass.depthAttachment.slice = map.sliceIndex
            pass.depthAttachment.loadAction = .load
            pass.depthAttachment.storeAction = .store
            guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else { continue }
            encoder.label = "layout edit \(index)"
            encoder.setViewport(map.viewport)
            encoder.setRenderPipelineState(armPipeline)
            encoder.setDepthStencilState(overlayDepthState)
            encoder.setCullMode(.none)
            let projection = drawable.computeProjection(convention: .rightUpBack, viewIndex: index)
            let viewProjection = projection * (head * view.transform).inverse
            for (texture, model) in quads {
                encoder.setFragmentTexture(texture, index: 0)
                var uniforms = ArmUniforms(mvp: viewProjection * model, uv: SIMD4(0, 0, 1, 1))
                encoder.setVertexBytes(&uniforms, length: MemoryLayout<ArmUniforms>.stride, index: 0)
                encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            }
            encoder.endEncoding()
        }
    }

    private func encodeArmHUD(
        drawable: LayerRenderer.Drawable,
        anchor: DeviceAnchor?,
        hands: [HandPose],
        shown: metal_d3d9_shown_frame,
        into commandBuffer: MTLCommandBuffer
    ) {
        guard let anchor, anchor.isTracked, let pointer = shown.arm_texture else { return }
        let texture = Unmanaged<MTLTexture>.fromOpaque(pointer).takeUnretainedValue()
        let shownSides = [shown.arm_shown.0, shown.arm_shown.1]
        let rects = [shown.arm_uv.0, shown.arm_uv.1].map { SIMD4<Float>($0.0, $0.1, $0.2, $0.3) }
        let tilt = SpatialInput.armHUDTilt * .pi / 180
        // About the grip's x: its up leans away from the player, its face
        // (+z, the player's way) turns up.
        var lean = matrix_identity_float4x4
        lean.columns.1 = [0, cos(tilt), -sin(tilt), 0]
        lean.columns.2 = [0, sin(tilt), cos(tilt), 0]
        var cards: [(model: simd_float4x4, uv: SIMD4<Float>)] = []
        for side in 0..<2 where shownSides[side] != 0 {
            let hand = hands[side]
            guard hand.valid, hand.state >= 2 else { continue }
            let uv = rects[side]
            let wide = (uv.z - uv.x) * Float(texture.width), high = (uv.w - uv.y) * Float(texture.height)
            guard wide > 0, high > 0 else { continue }
            let height = SpatialInput.armHUDSize
            let inward: Float = side == 0 ? Self.armHUDInward : -Self.armHUDInward
            let model = hand.originFromGrip * Self.translate(inward, SpatialInput.armHUDOffset, 0) * lean
                * Self.translate(0, height / 2, 0) * Self.scale(height * wide / high, height, 1)
            cards.append((model, uv))
        }
        guard !cards.isEmpty else { return }
        for (index, view) in drawable.views.enumerated() {
            let map = view.textureMap
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = drawable.colorTextures[map.textureIndex]
            pass.colorAttachments[0].slice = map.sliceIndex
            pass.colorAttachments[0].loadAction = .load
            pass.colorAttachments[0].storeAction = .store
            pass.depthAttachment.texture = drawable.depthTextures[map.textureIndex]
            pass.depthAttachment.slice = map.sliceIndex
            pass.depthAttachment.loadAction = .load
            pass.depthAttachment.storeAction = .store
            guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else { continue }
            encoder.label = "arm HUD \(index)"
            encoder.setViewport(map.viewport)
            encoder.setRenderPipelineState(armPipeline)
            encoder.setDepthStencilState(overlayDepthState)
            encoder.setCullMode(.none)
            encoder.setFragmentTexture(texture, index: 0)
            let projection = drawable.computeProjection(convention: .rightUpBack, viewIndex: index)
            let viewProjection = projection * (anchor.originFromAnchorTransform * view.transform).inverse
            for card in cards {
                var uniforms = ArmUniforms(mvp: viewProjection * card.model, uv: card.uv)
                encoder.setVertexBytes(&uniforms, length: MemoryLayout<ArmUniforms>.stride, index: 0)
                encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            }
            encoder.endEncoding()
        }
    }

    /// The orbs: health beside the left controller and blood beside the
    /// right, each a glass sphere in an iron cage with brass collars, cap
    /// and base, filled to its fraction. Each is ray traced in its own
    /// fragment shader from each eye's own position, so the two eyes see
    /// a real sphere at its depth: the glass refracts what is inside, the
    /// liquid is a plane through the sphere that tilts and settles as the
    /// orb moves and darkens with the thickness looked through. A porthole
    /// in the cage faces the head, framed by a riveted bezel, and on the
    /// glass there the game's own picture and label are printed, lit from
    /// behind: the weapon and its ammunition on the health orb, the
    /// discipline's icon and its name on the blood orb. One quad per orb
    /// per eye, bounded by the cage.
    ///
    /// In first person each orb sits `SpatialInput.orbOffset` out from its
    /// grip along the head's right, 7 cm above it, at the hand's
    /// predicted pose. In third person (`rt_host_third_person`) they ease
    /// to places beside the body that follow its facing, the head's yaw
    /// smoothed; the hands do the aiming there and are rarely in view.
    /// Drawn over the frame with no depth test, like the reticle. In first
    /// person a side whose controller has no position shows nothing; its
    /// gauge is back on the panel (`metal_d3d9_fullview_arm_hud`).
    private func encodeOrbs(
        drawable: LayerRenderer.Drawable,
        anchor: DeviceAnchor?,
        hands: [HandPose],
        shown: metal_d3d9_shown_frame,
        into commandBuffer: MTLCommandBuffer
    ) {
        guard let anchor, anchor.isTracked else { return }
        let vitals = Self.orbVitals()
        let head = anchor.originFromAnchorTransform
        let headPosition = SIMD3<Float>(head.columns.3.x, head.columns.3.y, head.columns.3.z)
        let right = simd_normalize(SIMD3<Float>(head.columns.0.x, head.columns.0.y, head.columns.0.z))
        let up = simd_normalize(SIMD3<Float>(head.columns.1.x, head.columns.1.y, head.columns.1.z))
        let layout = LayoutStore.shared.current
        let now = CACurrentMediaTime()
        let time = Float(now.truncatingRemainder(dividingBy: 1000))
        let dt = Float(min(max(now - orbLastTime, 0), 0.1))
        // Third person: towards the places by the body, back to the hands
        // out of it, over about a third of a second.
        let thirdPerson = rt_host_third_person() != 0
        orbThirdPerson += ((thirdPerson ? 1 : 0) - orbThirdPerson) * min(dt / 0.12, 1)
        // The body's facing: the head's yaw, followed slowly and only once
        // the head has turned past a glance's worth.
        let forwardFlat = SIMD2<Float>(-head.columns.2.x, -head.columns.2.z)
        if simd_length(forwardFlat) > 0.2 {
            let headYaw = atan2(forwardFlat.x, -forwardFlat.y)
            if let body = orbBodyYaw {
                var turn = headYaw - body
                turn = atan2(sin(turn), cos(turn))
                let beyond = abs(turn) > 0.35 ? turn - (turn > 0 ? 0.35 : -0.35) : 0
                orbBodyYaw = body + beyond * min(dt * 3, 1) + turn * min(dt * 0.4, 1)
            } else {
                orbBodyYaw = headYaw
            }
        }
        let yaw = orbBodyYaw ?? 0
        let bodyForward = SIMD3<Float>(sin(yaw), 0, -cos(yaw))
        let bodyRight = SIMD3<Float>(cos(yaw), 0, sin(yaw))
        let lift = SIMD3<Float>(0, 1, 0)
        let labelTexture = shown.arm_texture.map { Unmanaged<MTLTexture>.fromOpaque($0).takeUnretainedValue() }
        let labelShown = [shown.label_shown.0, shown.label_shown.1]
        let labelUV = [shown.label_uv.0, shown.label_uv.1].map { SIMD4<Float>($0.0, $0.1, $0.2, $0.3) }
        if !loggedOrbSetup {
            loggedOrbSetup = true
            let t = SpatialInput.orbTilt
            Self.log(
                "orbs: health on the \(SpatialInput.orbHealthHand == 1 ? "right" : "left") hand, turned left hand "
                    + "pitch \(t[0].x) yaw \(t[0].y) roll \(t[0].z), right hand pitch \(t[1].x) yaw \(t[1].y) "
                    + "roll \(t[1].z) degrees (pitch up turns the porthole up to the eyes, the cage's top leaning "
                    + "back); third person \(SpatialInput.orbThirdPersonSide * 100) cm out, "
                    + "\(SpatialInput.orbThirdPersonForward * 100) ahead, \(SpatialInput.orbThirdPersonHeight * 100) below"
            )
        }
        var orbs: [(uniforms: OrbUniforms, side: Int)] = []
        var centres: [SIMD3<Float>?] = [nil, nil]
        // side here is what the orb holds, 0 health and 1 blood; hand is
        // the controller it sits by (RT_ORB_HEALTH_SIDE).
        for side in 0..<2 {
            let handIndex = side == 0 ? SpatialInput.orbHealthHand : 1 - SpatialInput.orbHealthHand
            let hand = hands[handIndex]
            let level = side == 0 ? vitals.health : vitals.blood
            let handHeld = hand.valid && hand.state >= 2
            guard level >= 0, handHeld || orbThirdPerson > 0.5 else { continue }
            let out: Float = handIndex == 0 ? -1 : 1
            let orbLayout = layout.orbs[handIndex]
            let radius = orbLayout.size / 2
            // Beside the body in third person: out, up and ahead along the
            // body's facing (LayoutStore, the edit's).
            let body = HUDLayoutMath.Frame(origin: headPosition, axes: simd_float3x3(bodyRight, lift, bodyForward))
            let b = layout.bodyOrbs[handIndex]
            let placed = body.origin + bodyRight * (out * b.x) + lift * b.y + bodyForward * b.z
            var centre = placed
            // At the hand: from the grip along the head's right, up and back.
            let back = simd_normalize(SIMD3<Float>(head.columns.2.x, head.columns.2.y, head.columns.2.z))
            var handFrame: HUDLayoutMath.Frame?
            if handHeld {
                let grip = hand.originFromGrip.columns.3
                let frame = HUDLayoutMath.Frame(origin: SIMD3(grip.x, grip.y, grip.z), axes: simd_float3x3(right, up, back))
                handFrame = frame
                let o = orbLayout.offset
                let atHand = frame.origin + right * (out * o.x) + up * o.y + back * o.z
                centre = simd_mix(atHand, placed, SIMD3(repeating: orbThirdPerson))
            }
            centres[side] = centre
            // The porthole faces the head, turned about the orb's upright,
            // then by the orb's own turn; the shader works in that frame.
            var facing = headPosition - centre
            facing.y = 0
            facing = simd_length(facing) > 1e-4 ? simd_normalize(facing) : -bodyForward
            // Beside the body it has its own turn (the edit's); between the
            // two places it takes the nearer one's.
            let turn = orbThirdPerson > 0.5 ? layout.bodyTurn(handIndex) : orbLayout.turn
            let orient = Self.orbOrientation(facing: facing, tilt: turn, mirrored: handIndex == 0)
            // What the layout edit can pick up this frame.
            let world = simd_float4x4(
                SIMD4(orient.columns.0, 0), SIMD4(orient.columns.1, 0), SIMD4(orient.columns.2, 0), SIMD4(centre, 1)
            )
            if orbThirdPerson > 0.5 {
                editPieces[.bodyOrb(hand: handIndex)] = LayoutEditor.Placed(
                    world: world, radius: radius, anchor: body, facing: HUDLayoutMath.facingFrame(facing: facing)
                )
            } else if let handFrame {
                editPieces[.orb(hand: handIndex)] = LayoutEditor.Placed(
                    world: world, radius: radius, anchor: handFrame, facing: HUDLayoutMath.facingFrame(facing: facing)
                )
            }
            let toLocal = orient.transpose
            let tilt = orbTilt[side]
            let normal = toLocal * simd_normalize(SIMD3<Float>(tilt.x, 1, tilt.y))
            let light = toLocal * simd_normalize(SIMD3<Float>(-0.42, 0.82, 0.39))
            let ticks = SIMD4<Float>(
                Self.liquidHeight(0.25), Self.liquidHeight(0.5), Self.liquidHeight(0.75), handIndex == 0 ? -1 : 1
            )
            let colour: SIMD4<Float> = side == 0 ? [0.06, 0.30, 0.95, 1] : [0.70, 0.015, 0.03, 1]
            // The picture and the text, laid out inside the porthole
            // (OrbLayout): the picture cropped to what it shows and filling
            // the porthole, the text as the game drew its lines, set along
            // the porthole's lower edge.
            var pictureFit: OrbLayout.Picture?
            if let texture = picture[side] {
                let crop = orbReads.crop(side: side, of: pictureSource[side]) ?? SIMD4<Float>(0, 0, 1, 1)
                let wide = (crop.z - crop.x) * Float(texture.width), high = (crop.w - crop.y) * Float(texture.height)
                if wide > 0, high > 0 { pictureFit = OrbLayout.Picture(uv: crop, aspect: wide / high) }
            }
            var lines: [OrbLayout.Line] = []
            if labelShown[side] != 0, let labelTexture {
                let uv = labelUV[side]
                if let read = orbReads.lines(side: side, uv: uv) {
                    lines = read
                } else {
                    let wide = (uv.z - uv.x) * Float(labelTexture.width)
                    let high = (uv.w - uv.y) * Float(labelTexture.height)
                    if wide > 0, high > 0 { lines = [OrbLayout.Line(uv: uv, aspect: wide / high)] }
                }
            }
            let text = side == 0
                ? OrbLayout.TextSize(preferred: 0.22, minimum: 0.14) : OrbLayout.TextSize(preferred: 0.2, minimum: 0.11)
            // The power's name goes on the blood orb's nameplate, the
            // porthole keeping only the icon; the count stays on the arc.
            let plate = side == 1 ? OrbLayout.plate(lines: lines) : nil
            let layout = OrbLayout.layout(picture: pictureFit, lines: side == 1 ? [] : lines, text: text)
            var at = [SIMD4<Float>](repeating: .zero, count: 4), from = at
            var kinds = SIMD4<Float>.zero
            var k = 0
            if let rect = layout.picture {
                at[k] = SIMD4(rect.centre.x, rect.centre.y, rect.half.x, rect.half.y)
                from[k] = rect.uv
                kinds[k] = 1
                k += 1
            }
            for arc in layout.arcs.prefix(4 - k) {
                at[k] = SIMD4(arc.inner, arc.outer, arc.start, arc.end)
                from[k] = arc.uv
                // The count is printed bold; the name as drawn.
                kinds[k] = side == 0 ? 5 : 4
                k += 1
            }
            let uniforms = OrbUniforms(
                viewProjection: matrix_identity_float4x4, centre: SIMD4(centre, radius), eye: .zero,
                colour: colour, plane: SIMD4(normal, Self.liquidHeight(level)),
                misc: [level, time + Float(side) * 1.7, 0, 0],
                facing: SIMD4(0, 0, 1, Self.portholeCos),
                at0: at[0], at1: at[1], at2: at[2], at3: at[3],
                from0: from[0], from1: from[1], from2: from[2], from3: from[3], kinds: kinds,
                orient: simd_float4x4(
                    SIMD4(orient.columns.0, 0), SIMD4(orient.columns.1, 0), SIMD4(orient.columns.2, 0), [0, 0, 0, 1]
                ),
                light: SIMD4(light, 0), ticks: ticks, plate: .zero
            )
            orbs.append((uniforms, side))
            // The nameplate: a second draw of the same orb, bounded round
            // the plate, its lines as the prints.
            if let plate {
                var plated = uniforms
                plated.plate = SIMD4(plate.centreY, plate.half.x, plate.half.y, 1)
                var lineAt = [SIMD4<Float>](repeating: .zero, count: 4), lineFrom = lineAt
                var lineKinds = SIMD4<Float>.zero
                for (k, line) in plate.lines.prefix(4).enumerated() {
                    lineAt[k] = SIMD4(line.centre.x, line.centre.y, line.half.x, line.half.y)
                    lineFrom[k] = line.uv
                    lineKinds[k] = 6
                }
                (plated.at0, plated.at1, plated.at2, plated.at3) = (lineAt[0], lineAt[1], lineAt[2], lineAt[3])
                (plated.from0, plated.from1, plated.from2, plated.from3) = (lineFrom[0], lineFrom[1], lineFrom[2], lineFrom[3])
                plated.kinds = lineKinds
                orbs.append((plated, side))
            }
        }
        stepSlosh(centres: centres, now: now)
        guard !orbs.isEmpty else { return }
        for (index, view) in drawable.views.enumerated() {
            let map = view.textureMap
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = drawable.colorTextures[map.textureIndex]
            pass.colorAttachments[0].slice = map.sliceIndex
            pass.colorAttachments[0].loadAction = .load
            pass.colorAttachments[0].storeAction = .store
            pass.depthAttachment.texture = drawable.depthTextures[map.textureIndex]
            pass.depthAttachment.slice = map.sliceIndex
            pass.depthAttachment.loadAction = .load
            pass.depthAttachment.storeAction = .store
            guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else { continue }
            encoder.label = "orbs \(index)"
            encoder.setViewport(map.viewport)
            encoder.setDepthStencilState(SpatialInput.hands ? orbDepthState : overlayDepthState)
            encoder.setCullMode(.none)
            let projection = drawable.computeProjection(convention: .rightUpBack, viewIndex: index)
            let originFromView = anchor.originFromAnchorTransform * view.transform
            let viewProjection = projection * originFromView.inverse
            let eye = SIMD4<Float>(originFromView.columns.3.x, originFromView.columns.3.y, originFromView.columns.3.z, 1)
            encoder.setRenderPipelineState(SpatialInput.hands ? orbDepthPipeline : orbPipeline)
            encoder.setFragmentTexture(labelTexture ?? placeholder, index: 1)
            for orb in orbs {
                var uniforms = orb.uniforms
                uniforms.viewProjection = viewProjection
                uniforms.eye = eye
                encoder.setFragmentTexture(picture[orb.side] ?? placeholder, index: 0)
                encoder.setVertexBytes(&uniforms, length: MemoryLayout<OrbUniforms>.stride, index: 0)
                encoder.setFragmentBytes(&uniforms, length: MemoryLayout<OrbUniforms>.stride, index: 0)
                encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            }
            encoder.endEncoding()
        }
    }

    /// Our hands' renderer and models: the modelled gloves where the app
    /// carries them (`glove_right.usdz` and `glove_left.usdz`, in the
    /// bundle or the folder `RT_HANDS_MODEL` names), else the stand-in hand
    /// made in code, the left as the right's mirror. And the backend asked
    /// for the scene's depth, so a hand goes behind a wall.
    private func setUpGloves(colourFormat: MTLPixelFormat, depthFormat: MTLPixelFormat) {
        do {
            gloveRenderer = try HandRenderer(device: device, colourFormat: colourFormat, depthFormat: depthFormat)
        } catch {
            Self.log("hands: the renderer did not build: \(error)")
            return
        }
        func url(_ name: String) -> URL? {
            if let dir = ProcessInfo.processInfo.environment["RT_HANDS_MODEL"] {
                let u = URL(fileURLWithPath: dir).appendingPathComponent(name)
                if FileManager.default.fileExists(atPath: u.path) { return u }
            }
            return Bundle.main.url(forResource: name, withExtension: nil, subdirectory: "Hands")
                ?? Bundle.main.url(forResource: name, withExtension: nil)
        }
        let right = ProceduralHand.rig()
        gloveRigs = [right.mirrored(), right]
        let standIn = ProceduralHand.mesh()
        gloveMeshes = [
            HandMesh(device: device, mesh: ProceduralHand.mirrored(standIn), jointCount: right.count),
            HandMesh(device: device, mesh: standIn, jointCount: right.count),
        ]
        var source = "the stand-in made in code"
        for (side, name) in [(1, "glove_right.usdz"), (0, "glove_left.usdz")] {
            guard let u = url(name) else { continue }
            do {
                let model = try HandModelLoader.load(url: u, left: side == 0)
                guard model.rig.count <= HandRenderer.maxJoints, let mesh = HandMesh(device: device, model: model) else {
                    Self.log("hands: \(name) has \(model.rig.count) joints, more than \(HandRenderer.maxJoints); not used")
                    continue
                }
                gloveRigs[side] = model.rig
                gloveMeshes[side] = mesh
                glovePoseFiles[side] = model.poses
                let matched = HandJoint.allCases.filter { model.rig.joint[$0] != nil }.count
                source = "the model"
                Self.log(
                    "hands: \(name), \(model.mesh.vertexCount) vertices, \(matched) of \(HandJoint.allCases.count) "
                        + "joints matched, \(model.baseColourTexture != nil ? "textured" : "untextured"), "
                        + "\(model.poses?.poses.count ?? 0) poses of its own"
                )
            } catch {
                Self.log("hands: \(name) did not load: \(error)")
            }
        }
        metal_d3d9_fullview_scene_depth(1)
        Self.log(
            "hands: on, \(source); offset \(SpatialInput.handsOffset * 100) cm, turned \(SpatialInput.handsRotation) "
                + "degrees, scale \(SpatialInput.handsScale)\(SpatialInput.handsInThirdPerson ? ", in third person too" : "")"
        )
    }

    /// The hand at each tracked controller, in each view, depth tested and
    /// writing depth so the orbs sort against it; in full view hidden where
    /// the game drew something nearer (the scene's depth, as the frame was
    /// drawn and turned to now as its picture is), past what is in the hand.
    /// Posed by the controls: the trigger curls the index, the grip the
    /// other three, a thumb on a button lowers the thumb; and by what is
    /// held: the gun hand closes round a pistol's or a rifle's grip, or a
    /// hilt when a weapon shows no crosshair, the other hand round a
    /// rifle's fore end; in a menu both point.
    /// Turns a layout whose glove places were measured from the grips into
    /// places in the guns' frames, once, keeping each glove where it was in
    /// the room this frame; the built in places too, for a reset.
    static func reanchorGloves(grips: [simd_float4x4], guns: [simd_float4x4]) {
        LayoutStore.shared.update { l in
            guard l.gloveAnchor != "gun" else { return }
            let move = { (g: HUDLayout.Glove, side: Int) in
                HUDLayoutMath.reanchor(g, mirrored: side == 0, from: grips[side], to: guns[side])
            }
            l.gloves = (0..<2).map { move(l.gloves[$0], $0) }
            if let all = l.weaponGloves {
                l.weaponGloves = all.mapValues { pair in pair.count == 2 ? (0..<2).map { move(pair[$0], $0) } : pair }
            }
            let d = LayoutStore.defaults.gloves
            l.builtGloves = (0..<2).map { move(d[$0], $0) }
            l.gloveAnchor = "gun"
        }
        LayoutStore.shared.save()
        log("hands: the glove places now measured from the guns' frames, turned from the grips' once")
    }

    private func encodeGloves(
        drawable: LayerRenderer.Drawable,
        anchor: DeviceAnchor?,
        hands: [HandPose],
        shown: metal_d3d9_shown_frame,
        fullView: Bool,
        menu: Bool,
        into commandBuffer: MTLCommandBuffer
    ) {
        guard let renderer = gloveRenderer, let anchor, anchor.isTracked else { return }
        if rt_host_third_person() != 0, !SpatialInput.handsInThirdPerson, !menu { return }
        let now = CACurrentMediaTime()
        // Once a frame: the views share the poses.
        if now != gloveTime {
            let dt = Float(gloveTime == 0 ? 0 : min(max(now - gloveTime, 0), 0.1))
            gloveTime = now
            // A weapon is out when the game has a first person weapon model
            // (rt_host_weapon), or failing that the HUD's weapon picture: the
            // picture alone missed on A.'s run (2026-11-25), so the gun hand
            // stayed relaxed with a gun out.
            let modelName = LayoutStore.shared.weapon
            let hasModel = !modelName.isEmpty && !modelName.hasSuffix(".mdl")
            let weapon = hasModel || shown.weapon_texture != nil
            if !weapon || shown.crosshair_shown != 0 { gloveCrosshairSeen = now }
            var gun = HandHold.pistol
            if weapon, now - gloveCrosshairSeen > 0.4 {
                gun = .melee
            } else if let texture = picture[0], let crop = orbReads.crop(side: 0, of: pictureSource[0]) {
                let wide = (crop.z - crop.x) * Float(texture.width), high = (crop.w - crop.y) * Float(texture.height)
                if high > 0, wide / high > 2.2 { gun = .rifle }
            }
            let aim = SpatialInput.aimSide.rawValue
            // With a gun's own model out the gun hand's glove is left out by
            // default, and the gun floats in the hand as in many VR
            // shooters: a glove that does not truly hold it read as
            // clipping. RT_HANDS_GUN=1 keeps it on the gun to work on.
            gloveGunHidden = fullView && !menu && hasModel && gun != .melee && !Self.handsOnGun ? aim : -1
            HandAttach.shared.weaponClass = menu || !fullView || !weapon ? 0 : gun == .melee ? 3 : gun == .rifle ? 2 : 1
            for side in 0..<2 {
                let hold: HandHold
                if menu || !fullView {
                    hold = .point
                } else if !weapon {
                    hold = .nothing
                } else if side == aim {
                    hold = gun
                } else {
                    hold = gun == .rifle ? .support : .nothing
                }
                let (controls, _) = HandControls.shared.state(side: side)
                let held = LayoutStore.shared.current.glove(side, weapon: LayoutStore.shared.weapon)
                if let file = glovePoseFiles[side],
                   let pose = gloveBlenders[side].step(controls: controls, hold: hold, dt: dt, file: file, rig: gloveRigs[side],
                                                       gripCurl: held.gripCurl, thumbWrap: held.thumbWrap) {
                    glovePoses[side] = pose
                } else {
                    let shape = gloveBlenders[side].step(controls: controls, hold: hold, dt: dt)
                    glovePoses[side] = shape.pose(on: gloveRigs[side])
                }
            }
            if !gloveLogged {
                gloveLogged = true
                Self.log("hands: first drawn, \(fullView ? "full view" : "on the screen"), the gun hand holding \(gun)")
            }
        }
        // Each hand's place: its grip, then the fitting, mirrored for the
        // left hand.
        var draws: [(HandMesh, [simd_float4x4], simd_float4x4)] = []
        // The game's own hands (HandAttach), placed as the third person
        // aim's point is.
        if HandAttach.shared.enabled, fullView {
            let eyes = drawable.views.reduce(SIMD4<Float>(0, 0, 0, 0)) { $0 + $1.transform.columns.3 } / Float(max(drawable.views.count, 1))
            let mid = anchor.originFromAnchorTransform * SIMD4(eyes.x, eyes.y, eyes.z, 1)
            HandAttach.shared.update(shown: shown, mid: mid, drawn: Self.viewRotation(yaw: shown.pose_yaw_deg, pitch: shown.pose_pitch_deg), now: now)
        }
        let layout = LayoutStore.shared.current
        // The weapon in hand, so each weapon keeps its own glove places.
        var weaponName = [CChar](repeating: 0, count: 48)
        let weapon = rt_host_weapon(&weaponName, Int32(weaponName.count)) != 0 ? String(cString: weaponName) : ""
        if weapon != LayoutStore.shared.weapon {
            LayoutStore.shared.weapon = weapon
            Self.log("hands: the gloves' places for \(weapon.isEmpty ? "no weapon" : weapon)")
        }
        // Each glove is placed in the frame its hand's gun is drawn in (the
        // hand aim's), so it moves with the gun as one piece, roll and all.
        let head = anchor.originFromAnchorTransform
        let headYaw = atan2(head.columns.2.x, head.columns.2.z)
        let gunFrames: [simd_float4x4?] = (0..<2).map { side in
            guard side < hands.count, hands[side].valid, hands[side].state >= 2 else { return nil }
            return HUDLayoutMath.gunFrame(aim: hands[side].originFromAim, tiltDegrees: SpatialInput.aimTilt, headYaw: headYaw,
                                          rest: SpatialInput.handAimRest, unitsPerMetre: SpatialInput.unitsPerMetre)
        }
        if layout.gloveAnchor != "gun", let g0 = gunFrames[0], let g1 = gunFrames[1] {
            Self.reanchorGloves(grips: [hands[0].originFromGrip, hands[1].originFromGrip], guns: [g0, g1])
        }
        for hand in hands where hand.valid && hand.state >= 2 {
            let side = hand.side.rawValue
            guard let mesh = gloveMeshes[side], let pose = glovePoses[side], let gun = gunFrames[side] else { continue }
            if side == gloveGunHidden { continue }
            let rig = gloveRigs[side]
            // The fitting from the layout (the edit's), mirrored for the
            // left hand.
            let glove = layout.glove(side, weapon: weapon)
            let fit = HUDLayoutMath.gloveFit(glove, mirrored: side == 0)
            let scale = glove.scale
            editPieces[.glove(hand: side)] = LayoutEditor.Placed(
                world: gun * fit, radius: 0.07 * scale, anchor: HUDLayoutMath.Frame(gun)
            )
            var placed = gun * fit
            if let onGame = HandAttach.shared.attached(side: side, gunSide: SpatialInput.aimSide.rawValue, placed: placed, now: now) {
                placed = onGame
            }
            let model = placed * simd_float4x4(diagonal: SIMD4(scale, scale, scale, 1))
            draws.append((mesh, rig.palette(local: rig.local(for: pose)), model))
        }
        guard !draws.isEmpty else { return }
        let sceneTexture = fullView ? shown.scene_depth.map { Unmanaged<MTLTexture>.fromOpaque($0).takeUnretainedValue() } : nil
        var turn = matrix_identity_float4x4
        if shown.pose_valid != 0 {
            turn = Self.lateRotation(
                originFromDevice: anchor.originFromAnchorTransform, yaw: shown.pose_yaw_deg, pitch: shown.pose_pitch_deg
            )
        }
        var eyeTan = shown.eye_tan
        let tangents = withUnsafeBytes(of: &eyeTan) { Array($0.bindMemory(to: Float.self)) }
        let deviceFromOrigin = anchor.originFromAnchorTransform.inverse
        let xs = drawable.views.map { $0.transform.columns.3.x }
        let leftmost = xs.min() ?? 0
        for (index, view) in drawable.views.enumerated() {
            let map = view.textureMap
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = drawable.colorTextures[map.textureIndex]
            pass.colorAttachments[0].slice = map.sliceIndex
            pass.colorAttachments[0].loadAction = .load
            pass.colorAttachments[0].storeAction = .store
            pass.depthAttachment.texture = drawable.depthTextures[map.textureIndex]
            pass.depthAttachment.slice = map.sliceIndex
            pass.depthAttachment.loadAction = .load
            pass.depthAttachment.storeAction = .store
            guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else { continue }
            encoder.label = "hands \(index)"
            encoder.setViewport(map.viewport)
            let projection = drawable.computeProjection(convention: .rightUpBack, viewIndex: index)
            let originFromView = anchor.originFromAnchorTransform * view.transform
            let viewProjection = projection * originFromView.inverse
            let eye = SIMD3<Float>(originFromView.columns.3.x, originFromView.columns.3.y, originFromView.columns.3.z)
            var scene: SceneDepth?
            if let sceneTexture, tangents.count == 8 {
                let right = xs.count == 2 && view.transform.columns.3.x > leftmost
                // The room into the eye image's axes: to the device, from
                // this eye, turned back to how the frame was drawn.
                var fromEye = matrix_identity_float4x4
                fromEye.columns.3 = SIMD4(-view.transform.columns.3.xyz, 1)
                let t = Array(tangents[(right ? 4 : 0)..<(right ? 8 : 4)])
                scene = SceneDepth(
                    texture: sceneTexture, slice: right ? 1 : 0, imageFromOrigin: turn.transpose * fromEye * deviceFromOrigin,
                    tangents: SIMD4(t[0], t[1], t[2], t[3]), bias: Self.gloveOcclusionBias,
                    // With the gun's depth at its true distance, the gun is
                    // what hides the fingers wrapped behind its grip; only
                    // without it is the near scene taken as what is held.
                    ignoreNearer: shown.scene_depth_vm != 0 ? 0 : Self.gloveHeldReach
                )
            }
            for (mesh, palette, model) in draws {
                renderer.draw(
                    mesh, palette: palette, model: model, viewProjection: viewProjection, eye: eye,
                    scene: Self.gloveOcclusion ? scene : nil, encoder: encoder
                )
            }
            encoder.endEncoding()
        }
    }

    /// The scene hides a hand only where it is this much nearer than the
    /// hand, metres, so a hand flat on a wall does not flicker through it;
    /// and nothing nearer the eye than `gloveHeldReach` hides it, which is
    /// what the hand holds, drawn into the game's picture as well.
    /// `RT_HANDS_OCCLUDE=0` draws the hands over everything.
    private static let gloveOcclusion = ProcessInfo.processInfo.environment["RT_HANDS_OCCLUDE"] != "0"
    private static let gloveOcclusionBias: Float = 0.03
    private static let gloveHeldReach: Float = 0.12
    /// `RT_HANDS_GUN=1`: the gun hand's glove stays drawn on a gun.
    private static let handsOnGun: Bool = ProcessInfo.processInfo.environment["RT_HANDS_GUN"] == "1"
    /// The side whose glove is left out this frame (a gun in that hand), or -1.
    private var gloveGunHidden = -1

    /// The porthole's half angle about its facing, as a cosine: 52 degrees,
    /// just short of the collars at 59 degrees above and below the middle.
    private static let portholeCos: Float = cos(52 * .pi / 180)

    /// An orb's frame in the room: x its porthole's right as seen, y its
    /// upright, z its porthole's facing, turned by its tilt (degrees:
    /// pitch turns the porthole up, yaw outwards, roll clockwise as seen;
    /// yaw and roll mirrored for the left hand so both orbs tilt alike).
    private static func orbOrientation(facing: SIMD3<Float>, tilt: SIMD3<Float>, mirrored: Bool) -> simd_float3x3 {
        let up = SIMD3<Float>(0, 1, 0)
        let x = simd_normalize(simd_cross(up, facing))
        let base = simd_float3x3(x, up, facing)
        let sign: Float = mirrored ? -1 : 1
        let radians = tilt * (.pi / 180)
        let yaw = simd_quatf(angle: radians.y * sign, axis: [0, 1, 0])
        let pitch = simd_quatf(angle: -radians.x, axis: [1, 0, 0])
        let roll = simd_quatf(angle: -radians.z * sign, axis: [0, 0, 1])
        return base * simd_float3x3(yaw * pitch * roll)
    }

    /// The liquid's surface as a spring: the orb's acceleration (from its
    /// centre's last positions) pushes the surface's normal against it,
    /// as liquid leans back when its glass is pulled, and it settles to
    /// level within a second or so. Once a compositor frame.
    private func stepSlosh(centres: [SIMD3<Float>?], now: Double) {
        let dt = Float(now - orbLastTime)
        orbLastTime = now
        for side in 0..<2 {
            guard let centre = centres[side], dt > 0.001, dt < 0.1, let last = orbLastCentre[side] else {
                orbLastCentre[side] = centres[side]
                orbLastVelocity[side] = .zero
                orbTilt[side] = .zero
                orbTiltSpeed[side] = .zero
                continue
            }
            let velocity = (centre - last) / dt
            let acceleration = (velocity - orbLastVelocity[side]) / dt
            orbLastCentre[side] = centre
            orbLastVelocity[side] = velocity
            // Stiffness and damping for a slosh of about 1.4 Hz that dies
            // in a few swings; the push is the acceleration across, over g.
            let push = SIMD2<Float>(acceleration.x, acceleration.z) / 9.81
            let spring: SIMD2<Float> = orbTilt[side] * 80
            let damping: SIMD2<Float> = orbTiltSpeed[side] * 4.5
            let drive: SIMD2<Float> = simd_clamp(push, SIMD2<Float>(repeating: -6), SIMD2<Float>(repeating: 6)) * 80
            let pull: SIMD2<Float> = -spring - damping - drive
            orbTiltSpeed[side] += pull * dt
            orbTilt[side] += orbTiltSpeed[side] * dt
            let length = simd_length(orbTilt[side])
            if length > 0.7 { orbTilt[side] *= 0.7 / length }
        }
    }

    /// The liquid's surface height in the sphere's radii for a fill
    /// fraction: linear on the vertical diameter, so the surface moves as
    /// soon as anything is spent. By volume (the cap's), a full orb's
    /// surface barely moved for a large loss and the glass only showed
    /// when nearly empty (headset run, 2026-11-25).
    private static func liquidHeight(_ fraction: Float) -> Float {
        min(max(fraction, 0), 1) * 2 - 1
    }

    /// One orb's picture, scaled up once each time the game shows another:
    /// the game's own texture (draw.m, orb_note; the discipline's icon 128
    /// pixels square, the weapon's 128 by 64) taken to premultiplied
    /// pixels, then four times the pixels by a Lanczos filter with a light
    /// sharpen, and where MetalFX's spatial scaler runs on this device its
    /// colour is joined to the Lanczos alpha, which it does not treat.
    /// Mipmapped for the distance. Only in this command buffer and memory:
    /// nothing of the game's is kept anywhere else. A frame with no picture
    /// keeps the last one.
    private func refreshPicture(_ pointer: UnsafeMutableRawPointer?, side: Int, into commandBuffer: MTLCommandBuffer) {
        guard let pointer else { return }
        let source = Unmanaged<MTLTexture>.fromOpaque(pointer).takeUnretainedValue()
        if let last = pictureSource[side], last === source { return }
        let w = source.width, h = source.height, scale = 2
        func target(_ width: Int, _ height: Int, mipmapped: Bool, usage: MTLTextureUsage) -> MTLTexture? {
            let d = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .rgba8Unorm, width: width, height: height, mipmapped: mipmapped
            )
            d.usage = usage
            d.storageMode = .private
            return device.makeTexture(descriptor: d)
        }
        let rw: MTLTextureUsage = [.renderTarget, .shaderRead]
        guard let small = target(w, h, mipmapped: false, usage: rw),
              let lanczos = target(w * scale, h * scale, mipmapped: true, usage: rw)
        else { return }
        func draw(_ pipeline: MTLRenderPipelineState, into texture: MTLTexture, _ inputs: [MTLTexture], _ label: String) {
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = texture
            pass.colorAttachments[0].loadAction = .dontCare
            pass.colorAttachments[0].storeAction = .store
            guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else { return }
            encoder.label = label
            encoder.setRenderPipelineState(pipeline)
            for (i, input) in inputs.enumerated() { encoder.setFragmentTexture(input, index: i) }
            encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            encoder.endEncoding()
        }
        draw(iconDecodePipeline, into: small, [source], "picture decode")
        // What the picture shows, without its clear margins, read once.
        let reads = orbReads
        let identity = ObjectIdentifier(source)
        readAlpha(small, rect: SIMD4(0, 0, w, h), into: commandBuffer) { alpha, width, height in
            reads.setCrop(side: side, source: identity, crop: OrbReads.crop(alpha: alpha, width: width, height: height))
        }
        var result = lanczos
        var how = "Lanczos"
        if let scaler = pictureScaler(width: w, height: h, scale: scale),
           let fx = target(w * scale, h * scale, mipmapped: false, usage: scaler.outputTextureUsage),
           let joined = target(w * scale, h * scale, mipmapped: true, usage: rw),
           let plain = target(w * scale, h * scale, mipmapped: false, usage: rw) {
            draw(iconLanczosPipeline, into: plain, [small], "picture Lanczos")
            scaler.colorTexture = small
            scaler.outputTexture = fx
            scaler.inputContentWidth = w
            scaler.inputContentHeight = h
            scaler.encode(commandBuffer: commandBuffer)
            draw(iconMergePipeline, into: joined, [fx, plain], "picture merge")
            result = joined
            how = "MetalFX colour with Lanczos alpha"
        } else {
            draw(iconLanczosPipeline, into: lanczos, [small], "picture Lanczos")
        }
        if let blit = commandBuffer.makeBlitCommandEncoder() {
            blit.generateMipmaps(for: result)
            blit.endEncoding()
        }
        pictureSource[side] = source
        picture[side] = result
        Self.log(
            "orbs: the \(side == 0 ? "weapon's picture" : "discipline's icon") \(w)x\(h) scaled to "
                + "\(w * scale)x\(h * scale) (\(how))"
        )
    }

    /// Each side's label read for its lines when its place on the label
    /// texture changes and every 20 frames besides, since the count's
    /// figures change in place; one read in flight per side.
    private func readLabels(shown: metal_d3d9_shown_frame, into commandBuffer: MTLCommandBuffer) {
        guard let pointer = shown.arm_texture else { return }
        let texture = Unmanaged<MTLTexture>.fromOpaque(pointer).takeUnretainedValue()
        let shownSides = [shown.label_shown.0, shown.label_shown.1]
        let uvs = [shown.label_uv.0, shown.label_uv.1].map { SIMD4<Float>($0.0, $0.1, $0.2, $0.3) }
        for side in 0..<2 where shownSides[side] != 0 {
            let uv = uvs[side]
            guard orbReads.wantsRead(side: side, uv: uv, frame: framesDrawn) else { continue }
            let size = SIMD2<Float>(Float(texture.width), Float(texture.height))
            let x0 = Int((uv.x * size.x).rounded(.down)), y0 = Int((uv.y * size.y).rounded(.down))
            let x1 = Int((uv.z * size.x).rounded(.up)), y1 = Int((uv.w * size.y).rounded(.up))
            guard x1 > x0, y1 > y0 else { continue }
            let reads = orbReads
            readAlpha(texture, rect: SIMD4(x0, y0, x1 - x0, y1 - y0), into: commandBuffer) { alpha, width, height in
                let lines = OrbReads.lines(
                    alpha: alpha, width: width, height: height, origin: SIMD2(x0, y0), size: SIMD2(Int(size.x), Int(size.y))
                )
                reads.setLines(side: side, uv: uv, lines: lines)
            }
        }
    }

    /// A rectangle of a texture's alpha, read back: drawn to an 8 bit
    /// target, copied to shared memory, and handed to `done` on the
    /// command buffer's completion.
    private func readAlpha(
        _ texture: MTLTexture, rect: SIMD4<Int>, into commandBuffer: MTLCommandBuffer,
        done: @escaping @Sendable ([UInt8], Int, Int) -> Void
    ) {
        let width = rect.z, height = rect.w
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r8Unorm, width: width, height: height, mipmapped: false)
        d.usage = [.renderTarget]
        d.storageMode = .private
        guard let mask = device.makeTexture(descriptor: d),
              let buffer = device.makeBuffer(length: width * height, options: .storageModeShared)
        else { return }
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = mask
        pass.colorAttachments[0].loadAction = .dontCare
        pass.colorAttachments[0].storeAction = .store
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else { return }
        encoder.label = "alpha mask"
        encoder.setRenderPipelineState(alphaMaskPipeline)
        encoder.setFragmentTexture(texture, index: 0)
        var origin = SIMD2<UInt32>(UInt32(rect.x), UInt32(rect.y))
        encoder.setFragmentBytes(&origin, length: MemoryLayout<SIMD2<UInt32>>.stride, index: 0)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
        guard let blit = commandBuffer.makeBlitCommandEncoder() else { return }
        blit.copy(
            from: mask, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
            sourceSize: MTLSize(width: width, height: height, depth: 1), to: buffer, destinationOffset: 0,
            destinationBytesPerRow: width, destinationBytesPerImage: width * height
        )
        blit.endEncoding()
        commandBuffer.addCompletedHandler { _ in
            let bytes = Array(UnsafeBufferPointer(start: buffer.contents().assumingMemoryBound(to: UInt8.self), count: width * height))
            done(bytes, width, height)
        }
    }

    /// MetalFX's spatial scaler for one input size, made once; nil where
    /// the device has none, and the Lanczos filter does it all.
    private func pictureScaler(width: Int, height: Int, scale: Int) -> MTLFXSpatialScaler? {
        let key = SIMD2(width, height)
        if let made = pictureScalers[key] { return made }
        var scaler: MTLFXSpatialScaler?
        if MTLFXSpatialScalerDescriptor.supportsDevice(device) {
            let d = MTLFXSpatialScalerDescriptor()
            d.inputWidth = width
            d.inputHeight = height
            d.outputWidth = width * scale
            d.outputHeight = height * scale
            d.colorTextureFormat = .rgba8Unorm
            d.outputTextureFormat = .rgba8Unorm
            d.colorProcessingMode = .perceptual
            scaler = d.makeSpatialScaler(device: device)
        }
        if scaler == nil { Self.log("orbs: no MetalFX spatial scaler for \(width)x\(height); Lanczos alone") }
        pictureScalers[key] = scaler
        return scaler
    }

    /// Health and blood as fractions of full for the orbs, from the
    /// runtime's latest read of the game's HUD; -1 hides an orb, as when
    /// there is no player in the world.
    private static func orbVitals() -> (health: Float, blood: Float) {
        var health: Int32 = 0, healthMax: Int32 = 0, blood: Int32 = 0, bloodMax: Int32 = 0
        guard rt_host_vitals(&health, &healthMax, &blood, &bloodMax) != 0 else { return (-1, -1) }
        func fraction(_ v: Int32, _ full: Int32) -> Float {
            full > 0 ? min(max(Float(v) / Float(full), 0), 1) : -1
        }
        return (fraction(health, healthMax), fraction(blood, bloodMax))
    }

    /// The ring at rest, in the shape's units, and the gauges' radius at
    /// rest (1.6 times it).
    static let reticleRingRest: Float = 0.7
    static let reticleGaugeRest: Float = 1.12
    /// The shape's unit as a tangent off its centre: the quad is 0.022 of
    /// its distance across its half and the shape's 1 is half of that.
    static let reticleUnitTan: Float = 0.011

    /// The ring follows the weapon's spread as the game's own crosshair
    /// does: wider running, tighter still, and with skill. The game's
    /// accuracy ring (draw.m, crosshair_note; only the ring, not its dot or
    /// its settled mark) has a half extent in its NDC
    /// that grows and shrinks with the spread; on full view's panel that is
    /// a tangent of half times the panel's, and the ring is put at that
    /// tangent, so it opens as far as the game's would on the same panel.
    /// Eased over a few frames, since the game's steps are coarse. A frame
    /// with no ring (no weapon, melee, a menu, third person, where the game
    /// draws no ring) eases back to the resting size.
    private func reticleRing(shown: metal_d3d9_shown_frame) -> Float {
        var target = Self.reticleRingRest
        if shown.crosshair_shown != 0, shown.panel.0 > 0 {
            // The ring's half, or with no ring drawn (the game shows its
            // settled mark alone once the aim has steadied) the last ring
            // held and eased down to the tightest seen over about half a
            // second, rather than snapped there, so a frame without a
            // ring in the middle of a recoil does not close it.
            var half = shown.crosshair_half
            if half > 0 {
                tightestRingHalf = min(tightestRingHalf, half)
                heldRingHalf = half
            } else {
                heldRingHalf += (tightestRingHalf - heldRingHalf) * 0.06
                half = heldRingHalf
            }
            target = min(max(half * shown.panel.0 / Self.reticleUnitTan, 0.35), 40)
        }
        reticleRing += (target - reticleRing) * 0.35
        return reticleRing
    }

    /// Which way the reticle went and whether the game drew a crosshair,
    /// in the log when either changes, the first `reticleLogLimit` times,
    /// so a device run shows the decision; the ring's size rides along as
    /// data, since it changes with every shot and would spend the lines.
    private func noteReticle(_ branch: String, thirdPerson: Bool, shown: metal_d3d9_shown_frame) {
        guard reticleLogged < Self.reticleLogLimit else { return }
        let key = "\(branch) \(shown.crosshair_shown) \(shown.aim_shown) \(shown.aim_hit)"
        guard key != reticleLastKey else { return }
        reticleLastKey = key
        reticleLogged += 1
        Self.log(
            String(
                format: "reticle: %@; third person %d, game crosshair %d at NDC %.4f %.4f half %.4f, aim %d hit %d "
                    + "at tangents %.4f %.4f %.2f m (t %.1f units), panel %.3f by %.3f, HUD plane %.2f m, pose %d yaw "
                    + "%.1f pitch %.1f",
                branch, thirdPerson ? 1 : 0, shown.crosshair_shown, Double(shown.crosshair_ndc.0),
                Double(shown.crosshair_ndc.1), Double(shown.crosshair_half), shown.aim_shown, shown.aim_hit,
                Double(shown.aim_tan.0), Double(shown.aim_tan.1), Double(shown.aim_m), Double(shown.aim_t),
                Double(shown.panel.0), Double(shown.panel.1), Double(shown.hud_m), shown.pose_valid,
                Double(shown.pose_yaw_deg), Double(shown.pose_pitch_deg)
            )
        )
    }

    private static let reticleLogLimit = 20

    /// How far each card sits towards the body from its grip's centre.
    private static let armHUDInward: Float = 0.02

    /// The health and blood gauges' fill, as fractions of full, from the
    /// runtime's latest read of the game's HUD; -1 hides a gauge, as when
    /// there is no player in the world or the gauges are off.
    private static func reticleGauges() -> (health: Float, blood: Float) {
        guard SpatialInput.reticleGauges else { return (-1, -1) }
        var health: Int32 = 0, healthMax: Int32 = 0, blood: Int32 = 0, bloodMax: Int32 = 0
        guard rt_host_vitals(&health, &healthMax, &blood, &bloodMax) != 0 else { return (-1, -1) }
        func fraction(_ v: Int32, _ full: Int32) -> Float {
            full > 0 ? min(max(Float(v) / Float(full), 0), 1) : -1
        }
        return (fraction(health, healthMax), fraction(blood, bloodMax))
    }

    private static func scale(_ x: Float, _ y: Float, _ z: Float) -> simd_float4x4 {
        simd_float4x4(diagonal: [x, y, z, 1])
    }

    private static func translate(_ x: Float, _ y: Float, _ z: Float) -> simd_float4x4 {
        var m = matrix_identity_float4x4
        m.columns.3 = [x, y, z, 1]
        return m
    }

    /// Both views' tangents and both eyes' positions in the device's space,
    /// left first, to the backend for full view when they change: the
    /// game's field and each eye's projection and offset come from them.
    private func publishViews(of drawable: LayerRenderer.Drawable) {
        guard drawable.views.count == 2 else { return }
        let order = drawable.views.indices.sorted {
            drawable.views[$0].transform.columns.3.x < drawable.views[$1].transform.columns.3.x
        }
        var tangents: [Float] = []
        var eyes: [Float] = []
        for index in order {
            var p = drawable.computeProjection(convention: .rightUpBack, viewIndex: index)
            var t = [Float](repeating: 0, count: 4)
            withUnsafeBytes(of: &p) { bytes in
                mtl_stereo_tangents(bytes.bindMemory(to: Float.self).baseAddress, &t)
            }
            tangents += t
            let c = drawable.views[index].transform.columns.3
            eyes += [c.x, c.y, c.z]
        }
        let all = tangents + eyes
        guard all != fullViewViews else { return }
        fullViewViews = all
        metal_d3d9_fullview_set_views(tangents, eyes)
    }

    /// Each view's projection, as tangents and as the matrix, its transform
    /// in the device's space, and its viewport and texture: what the eye
    /// planes are drawn through, for checking the stereo math against the
    /// real device rather than sample values.
    private func logViews(of drawable: LayerRenderer.Drawable) {
        for (index, view) in drawable.views.enumerated() {
            var p = drawable.computeProjection(convention: .rightUpBack, viewIndex: index)
            var t = [Float](repeating: 0, count: 4)
            withUnsafeBytes(of: &p) { bytes in
                mtl_stereo_tangents(bytes.bindMemory(to: Float.self).baseAddress, &t)
            }
            let m = view.transform
            let vp = view.textureMap.viewport
            let texture = drawable.colorTextures[view.textureMap.textureIndex]
            func row(_ c: SIMD4<Float>) -> String { "(\(c.x), \(c.y), \(c.z), \(c.w))" }
            Self.log(
                "view \(index): tangents left \(t[0]) right \(t[1]) down \(t[2]) up \(t[3]) "
                    + "(\(Self.degrees(t[0]) + Self.degrees(t[1])) by \(Self.degrees(t[2]) + Self.degrees(t[3])) degrees); "
                    + "projection columns \(row(p.columns.0)) \(row(p.columns.1)) \(row(p.columns.2)) \(row(p.columns.3)); "
                    + "device from view columns \(row(m.columns.0)) \(row(m.columns.1)) \(row(m.columns.2)) "
                    + "\(row(m.columns.3)); viewport \(vp.originX),\(vp.originY) \(vp.width)x\(vp.height) of "
                    + "\(texture.width)x\(texture.height) slice \(view.textureMap.sliceIndex)"
            )
        }
    }

    private static func degrees(_ tangent: Float) -> Float {
        atan(tangent) * 180 / .pi
    }

    private func sourceTexture(for shown: metal_d3d9_shown_frame) -> MTLTexture? {
        guard let pointer = shown.texture else { return nil }
        let texture = Unmanaged<MTLTexture>.fromOpaque(pointer).takeUnretainedValue()
        if !checkedDevice {
            checkedDevice = true
            let same = texture.device === device && renderer.device === device
            Self.log("first game frame \(shown.width)x\(shown.height); game, layer and compositor on one device: \(same)")
        }
        let key = ObjectIdentifier(texture)
        if let view = srgbViews[key] { return view }
        // A slot remade after a Reset gets a new identity; old views go.
        if srgbViews.count >= 8 { srgbViews.removeAll() }
        let view = texture.makeTextureView(pixelFormat: .bgra8Unorm_srgb)
        srgbViews[key] = view
        return view
    }

    /// The guest's cursor on the screen, in texture coordinates, with z set
    /// when it is shown: the guest's cursor is a system cursor in a window,
    /// which an immersive space does not have.
    /// A menu's cursor goes where the aiming hand's ray meets the menu's
    /// screen (the other hand's when that one is not tracked or misses),
    /// in place of the stick: the screen stands in the room on the anchor,
    /// `distance` ahead along its forward, `2 tan distance` across.
    private func pointCursor(hands: [HandPose], anchor: [Float], tanX: Float, tanY: Float, distance: Float) {
        guard anchor.count == 16, tanX > 0, tanY > 0, distance > 0 else { return }
        let right = SIMD3(anchor[0], anchor[1], anchor[2])
        let up = SIMD3(anchor[4], anchor[5], anchor[6])
        let back = SIMD3(anchor[8], anchor[9], anchor[10])
        let centre = SIMD3(anchor[12], anchor[13], anchor[14]) - back * distance
        let first = SpatialInput.aimSide
        for side in [first, first == .left ? HandPose.Side.right : .left] {
            let hand = hands[side.rawValue]
            guard hand.valid else { continue }
            let aim = hand.originFromAim
            let from = SIMD3(aim.columns.3.x, aim.columns.3.y, aim.columns.3.z)
            let dir = -SIMD3(aim.columns.2.x, aim.columns.2.y, aim.columns.2.z)
            let den = simd_dot(dir, back)
            guard abs(den) > 1e-4 else { continue }
            let t = simd_dot(centre - from, back) / den
            guard t > 0 else { continue }
            let rel = from + dir * t - centre
            let u = simd_dot(rel, right) / (2 * tanX * distance) + 0.5
            let v = 0.5 - simd_dot(rel, up) / (2 * tanY * distance)
            guard u >= 0, u <= 1, v >= 0, v <= 1 else { continue }
            rt_host_cursor_to(u, v)
            return
        }
    }

    private static func guestCursor() -> SIMD4<Float> {
        var u: Float = 0
        var v: Float = 0
        guard rt_host_cursor(&u, &v) != 0 else { return .zero }
        return SIMD4(u, v, 1, 0.008)
    }

    private static func placeScreen(head: simd_float4x4) -> [Float] {
        var head = head
        var model = [Float](repeating: 0, count: 16)
        withUnsafeBytes(of: &head) { bytes in
            let h = bytes.bindMemory(to: Float.self)
            mtl_screen_place(h.baseAddress, screenDistance, screenWidth, screenAspect, &model)
        }
        log(
            "screen placed at (\(model[12]), \(model[13]), \(model[14])), "
                + "\(screenWidth) m wide, \(screenDistance) m ahead of the head"
        )
        return model
    }

    /// The tested C math (immersive.h), so what runs is what was checked.
    private static func planeMVP(
        projection: simd_float4x4, deviceFromView: simd_float4x4, turn: simd_float4x4, tanX: Float, tanY: Float,
        distance: Float
    ) -> simd_float4x4 {
        var projection = projection
        var deviceFromView = deviceFromView
        var turn = turn
        var out = simd_float4x4()
        withUnsafeBytes(of: &projection) { p in
            withUnsafeBytes(of: &deviceFromView) { v in
                withUnsafeBytes(of: &turn) { t in
                    withUnsafeMutableBytes(of: &out) { o in
                        mtl_stereo_plane_mvp_turned(
                            p.bindMemory(to: Float.self).baseAddress,
                            v.bindMemory(to: Float.self).baseAddress,
                            t.bindMemory(to: Float.self).baseAddress,
                            tanX, tanY, distance,
                            o.bindMemory(to: Float.self).baseAddress
                        )
                    }
                }
            }
        }
        return out
    }

    private static func fullViewMVP(
        projection: simd_float4x4, deviceFromView: simd_float4x4, turn: simd_float4x4, tangents: [Float], distance: Float
    ) -> simd_float4x4 {
        var projection = projection
        var deviceFromView = deviceFromView
        var turn = turn
        var out = simd_float4x4()
        withUnsafeBytes(of: &projection) { p in
            withUnsafeBytes(of: &deviceFromView) { v in
                withUnsafeBytes(of: &turn) { t in
                    withUnsafeMutableBytes(of: &out) { o in
                        mtl_fullview_plane_mvp(
                            p.bindMemory(to: Float.self).baseAddress,
                            v.bindMemory(to: Float.self).baseAddress,
                            t.bindMemory(to: Float.self).baseAddress,
                            tangents, distance,
                            o.bindMemory(to: Float.self).baseAddress
                        )
                    }
                }
            }
        }
        return out
    }

    private static func stereoAnchor(head: simd_float4x4) -> [Float] {
        var head = head
        var out = [Float](repeating: 0, count: 16)
        withUnsafeBytes(of: &head) { h in
            mtl_stereo_anchor(h.bindMemory(to: Float.self).baseAddress, &out)
        }
        return out
    }

    private static func fixedPlaneMVP(
        projection: simd_float4x4, deviceFromView: simd_float4x4, originFromDevice: simd_float4x4, anchor: [Float],
        tanX: Float, tanY: Float, distance: Float
    ) -> simd_float4x4 {
        var projection = projection
        var deviceFromView = deviceFromView
        var originFromDevice = originFromDevice
        var out = simd_float4x4()
        withUnsafeBytes(of: &projection) { p in
            withUnsafeBytes(of: &deviceFromView) { v in
                withUnsafeBytes(of: &originFromDevice) { d in
                    withUnsafeMutableBytes(of: &out) { o in
                        mtl_stereo_plane_mvp_fixed(
                            p.bindMemory(to: Float.self).baseAddress,
                            v.bindMemory(to: Float.self).baseAddress,
                            d.bindMemory(to: Float.self).baseAddress,
                            anchor,
                            tanX, tanY, distance,
                            o.bindMemory(to: Float.self).baseAddress
                        )
                    }
                }
            }
        }
        return out
    }

    private static func lateRotation(originFromDevice: simd_float4x4, yaw: Float, pitch: Float) -> simd_float4x4 {
        var m = originFromDevice
        var out = simd_float4x4()
        withUnsafeBytes(of: &m) { a in
            withUnsafeMutableBytes(of: &out) { o in
                mtl_late_rotation(a.bindMemory(to: Float.self).baseAddress, yaw, pitch, o.bindMemory(to: Float.self).baseAddress)
            }
        }
        return out
    }

    /// The orientation a frame was drawn with, yaw and pitch with no roll
    /// (`mtl_view_rotation`): the drawn half of `lateRotation`'s turn.
    private static func viewRotation(yaw: Float, pitch: Float) -> simd_float4x4 {
        var out = simd_float4x4()
        withUnsafeMutableBytes(of: &out) { o in
            mtl_view_rotation(yaw, pitch, o.bindMemory(to: Float.self).baseAddress)
        }
        return out
    }

    private static func rotationDegrees(_ r: simd_float4x4) -> Float {
        var r = r
        return withUnsafeBytes(of: &r) { mtl_rotation_angle_deg($0.bindMemory(to: Float.self).baseAddress) }
    }

    private static func mvp(projection: simd_float4x4, originFromView: simd_float4x4, model: [Float]) -> simd_float4x4 {
        var projection = projection
        var originFromView = originFromView
        var out = simd_float4x4()
        withUnsafeBytes(of: &projection) { p in
            withUnsafeBytes(of: &originFromView) { v in
                withUnsafeMutableBytes(of: &out) { o in
                    mtl_screen_mvp(
                        p.bindMemory(to: Float.self).baseAddress,
                        v.bindMemory(to: Float.self).baseAddress,
                        model,
                        o.bindMemory(to: Float.self).baseAddress
                    )
                }
            }
        }
        return out
    }

    private static func name(of layout: LayerRenderer.Layout) -> String {
        switch layout {
        case .dedicated: "dedicated"
        case .shared: "shared"
        case .layered: "layered"
        @unknown default: "unknown (\(layout.rawValue))"
        }
    }

    static func log(_ message: String) {
        metal_d3d9_immersive_log(message)
    }

    private enum CompositorError: Error {
        case noDevice
    }

    /// The quad is drawn from its vertex index as a four-vertex strip, so
    /// there is no vertex buffer. uv (0, 0) is the back buffer's top left.
    private static let shaderSource = """
        #include <metal_stdlib>
        using namespace metal;

        struct Uniforms {
            float4x4 mvp;
            float4 cursor; // u, v, shown, radius
        };

        struct VertexOut {
            float4 position [[position]];
            float2 uv;
        };

        vertex VertexOut screen_vertex(uint vid [[vertex_id]], constant Uniforms &u [[buffer(0)]]) {
            float2 corner = float2((vid & 1) ? 0.5 : -0.5, (vid & 2) ? -0.5 : 0.5);
            VertexOut out;
            out.position = u.mvp * float4(corner, 0.0, 1.0);
            out.uv = float2(corner.x + 0.5, 0.5 - corner.y);
            return out;
        }

        fragment float4 screen_fragment(VertexOut in [[stage_in]], texture2d<float> tex [[texture(0)]],
                                        constant Uniforms &u [[buffer(0)]]) {
            constexpr sampler s(filter::linear, address::clamp_to_edge);
            float3 c = tex.sample(s, in.uv).rgb;
            if (u.cursor.z > 0.0) {
                // A ring with a dark edge, round on the 4:3 screen.
                float r = length((in.uv - u.cursor.xy) * float2(4.0 / 3.0, 1.0));
                float w = u.cursor.w;
                if (r < w * 1.35 && r > w * 0.55) c = r < w ? float3(1.0) : float3(0.0);
            }
            return float4(c, 1.0);
        }

        struct StereoUniforms {
            float4x4 mvp;
            float4 cursor; // u, v, shown, radius
            uint slice;
            float edge; // full view: how far in from the image's border it fades to black, in uv
            float dim; // full view: brightness, below 1 for the world held behind a menu
            float pad2;
        };

        // The eye's plane, uv (0, 0) the eye image's top left.
        vertex VertexOut stereo_vertex(uint vid [[vertex_id]], constant StereoUniforms &u [[buffer(0)]]) {
            float2 corner = float2((vid & 1) ? 0.5 : -0.5, (vid & 2) ? -0.5 : 0.5);
            VertexOut out;
            out.position = u.mvp * float4(corner, 0.0, 1.0);
            out.uv = float2(corner.x + 0.5, 0.5 - corner.y);
            return out;
        }

        // The cursor is a system cursor in the guest, drawn here in each
        // eye where the HUD plane puts it.
        fragment float4 stereo_fragment(VertexOut in [[stage_in]], texture2d_array<float> tex [[texture(0)]],
                                        constant StereoUniforms &u [[buffer(0)]]) {
            constexpr sampler s(filter::linear, address::clamp_to_edge);
            float3 c = tex.sample(s, in.uv, u.slice).rgb;
            if (u.dim > 0.0) c *= u.dim;
            if (u.cursor.z > 0.0) {
                float r = length((in.uv - u.cursor.xy) * float2(4.0 / 3.0, 1.0));
                float w = u.cursor.w;
                if (r < w * 1.35 && r > w * 0.55) c = r < w ? float3(1.0) : float3(0.0);
            }
            if (u.edge > 0.0) {
                float2 d = min(in.uv, 1.0 - in.uv);
                c *= smoothstep(0.0, u.edge, min(d.x, d.y));
            }
            return float4(c, 1.0);
        }

        struct HandUniforms {
            float4x4 mvp;
            float4 colour;
        };

        struct HandOut {
            float4 position [[position]];
            float shade;
        };

        // A unit cube centred on the origin from its vertex index, twelve
        // triangles; each face shaded a little differently so the box reads
        // as solid with no lighting.
        vertex HandOut hand_vertex(uint vid [[vertex_id]], constant HandUniforms &u [[buffer(0)]]) {
            const uchar faces[6][4] = {{0, 1, 3, 2}, {4, 6, 7, 5}, {0, 4, 5, 1}, {2, 3, 7, 6}, {0, 2, 6, 4}, {1, 5, 7, 3}};
            const uchar order[6] = {0, 1, 2, 0, 2, 3};
            uint face = vid / 6;
            uint corner = faces[face][order[vid % 6]];
            float3 p = float3((corner & 1) ? 0.5 : -0.5, (corner & 2) ? 0.5 : -0.5, (corner & 4) ? 0.5 : -0.5);
            HandOut out;
            out.position = u.mvp * float4(p, 1.0);
            out.shade = 0.7 + 0.06 * float(face);
            return out;
        }

        fragment float4 hand_fragment(HandOut in [[stage_in]], constant HandUniforms &u [[buffer(0)]]) {
            return float4(u.colour.rgb * in.shade, 1.0);
        }

        struct ReticleUniforms {
            float4x4 mvp;
            float4 colour;
            int style;
            float health;
            float blood;
            float extent;
            float ring;
            float gauge_radius;
        };

        struct ReticleOut {
            float4 position [[position]];
            float2 p;
        };

        // A square facing the head, p from -extent to extent across it.
        // The reticle's shape is drawn in p, so a larger square for the
        // gauges leaves it the size it was.
        vertex ReticleOut reticle_vertex(uint vid [[vertex_id]], constant ReticleUniforms &u [[buffer(0)]]) {
            float2 c = float2((vid & 1) ? 1.0 : -1.0, (vid & 2) ? -1.0 : 1.0) * u.extent;
            ReticleOut out;
            out.position = u.mvp * float4(c * 0.5, 0.0, 1.0);
            out.p = c;
            return out;
        }

        // One gauge: an arc of the ring at radius radius (1.12 at rest, 1.6
        // times the ring's, moving out as the ring widens), 120 degrees about angle 0 of the side's own angle,
        // filled from its lower end up to fraction f and dim beyond. Edges
        // are smoothed over a pixel as the reticle's are, the ends and the
        // fill's edge measured along the arc. Premultiplied out.
        float4 reticle_gauge(float2 p, float angle, float f, float3 colour, float px, float line, float radius) {
            if (f < 0.0) return float4(0.0);
            float half_span = M_PI_F / 3.0;
            float r = length(p);
            float band = 1.0 - smoothstep(line - px, line + px, abs(r - radius));
            float ends = smoothstep(-px, px, radius * (half_span - abs(angle)));
            float top = -half_span + clamp(f, 0.0, 1.0) * 2.0 * half_span;
            float full = smoothstep(-px, px, radius * (top - angle));
            float a = band * ends * mix(0.22, 0.85, full);
            return float4(colour * a, a);
        }

        // The shape as coverage from distances, smoothed over a pixel:
        // 0 cross, 1 circle and dot, 2 dot, 3 circle, 4 cross and dot.
        // The cross leaves a gap at the middle so what it sits on shows.
        fragment float4 reticle_fragment(ReticleOut in [[stage_in]], constant ReticleUniforms &u [[buffer(0)]]) {
            float2 p = in.p;
            float px = max(fwidth(p.x), fwidth(p.y));
            float r = length(p);
            float line = 0.07;
            // The ring at the spread's radius; the cross's arms move out
            // with it, the dot stays.
            float ring = 1.0 - smoothstep(line - px, line + px, abs(r - u.ring));
            float spread = u.ring / 0.7;
            float dot = 1.0 - smoothstep(0.14 - px, 0.14 + px, r);
            float arms = max(1.0 - smoothstep(line - px, line + px, abs(p.x)),
                             1.0 - smoothstep(line - px, line + px, abs(p.y)));
            float span = smoothstep(0.28 * spread - px, 0.28 * spread + px, r)
                * (1.0 - smoothstep(0.95 * spread - px, 0.95 * spread + px, r));
            float cross = arms * span;
            float a;
            switch (u.style) {
                case 1: a = max(ring, dot); break;
                case 2: a = dot; break;
                case 3: a = ring; break;
                case 4: a = max(cross, dot); break;
                default: a = cross; break;
            }
            a *= u.colour.a;
            float4 mark = float4(u.colour.rgb * a, a);
            // Health on the left, its angle measured from straight left so
            // both sides fill from below; blood on the right.
            float4 health = reticle_gauge(p, atan2(p.y, -p.x), u.health, float3(0.3, 0.72, 0.95), px, line, u.gauge_radius);
            float4 blood = reticle_gauge(p, atan2(p.y, p.x), u.blood, float3(0.72, 0.04, 0.07), px, line, u.gauge_radius);
            float4 gauges = health + blood;
            return mark + gauges * (1.0 - mark.a);
        }

        struct ArmUniforms {
            float4x4 mvp;
            float4 uv; // the side's part of the cards: u0, v0, u1, v1
        };

        struct ArmOut {
            float4 position [[position]];
            float2 uv;
        };

        vertex ArmOut arm_vertex(uint vid [[vertex_id]], constant ArmUniforms &u [[buffer(0)]]) {
            float2 corner = float2((vid & 1) ? 0.5 : -0.5, (vid & 2) ? -0.5 : 0.5);
            ArmOut out;
            out.position = u.mvp * float4(corner, 0.0, 1.0);
            out.uv = mix(u.uv.xy, u.uv.zw, float2(corner.x + 0.5, 0.5 - corner.y));
            return out;
        }

        float3 arm_linear(float3 c) {
            return select(pow((c + 0.055) / 1.055, 2.4), c / 12.92, c <= 0.04045);
        }

        // The cards hold the game's gamma-encoded colour premultiplied by
        // its alpha, and the drawable blends in linear: the colour is taken
        // back to straight, decoded and premultiplied again, so a gauge's
        // soft edge is not darkened. An additive draw leaves colour with no
        // alpha, which is decoded as it is.
        fragment float4 arm_fragment(ArmOut in [[stage_in]], texture2d<float> tex [[texture(0)]]) {
            constexpr sampler s(filter::linear, mip_filter::linear, address::clamp_to_edge);
            float4 c = tex.sample(s, in.uv);
            float3 rgb = c.a > 0.004 ? arm_linear(min(c.rgb / c.a, 1.0)) * c.a : arm_linear(c.rgb);
            return float4(rgb, c.a);
        }

        struct OrbUniforms {
            float4x4 viewProjection;
            float4 centre; // the orb's centre in the room, w its glass radius in metres
            float4 eye;    // this view's eye in the room
            float4 colour; // the liquid's colour, linear
            float4 plane;  // the liquid's surface: its normal (up, tilted), w its height in radii
            float4 misc;   // x the fill fraction, y seconds, z 1 with a picture, w 1 with a label
            float4 facing; // the porthole's axis, level and towards the head; w its half angle's cosine
            float4 at[4];   // each print's centre and half size on the porthole, in the glass's radii
            float4 from[4]; // and its rectangle on its texture: u0, v0, u1, v1
            float4 kinds;   // each print's: 0 none, 1 the picture (at: centre, half), 4 a line on an arc
                            // (at: inner and outer radius, start and end angle), 5 one printed bold
            float4x4 orient; // the orb's frame in the room (upright, porthole on +z); the rest is in it
            float4 light;    // the key light, in the orb's frame
            float4 ticks;    // the surface's heights at a quarter, half and three quarters; w the ticks' side
            float4 plate;    // the nameplate: its centre's height, half width and half height in radii;
                             // w 1 when this draw is the plate (kinds 6: its lines, at: centre, half on it)
        };

        struct OrbOut {
            float4 position [[position]];
            float3 world;
        };

        // The orb's bound: the cage reaches 1.32 radii from the centre. The
        // quad stands across the eye's line to the centre, just covering the
        // bound's silhouette from this eye, so each eye gets its own.
        constant float orb_bound = 1.32;

        // The nameplate's hangers run from the base's foot down to it.
        constant float plate_hang = -1.12;
        constant float plate_z = 0.34;

        // The nameplate's bound in the orb's frame: its centre and radius.
        float4 orb_plate_bound(float4 plate) {
            float top = plate_hang, bottom = plate.x - plate.z;
            float3 c = float3(0.0, (top + bottom) * 0.5, plate_z * 0.8);
            return float4(c, length(float3(plate.y, (top - bottom) * 0.5, 0.12)) + 0.04);
        }

        vertex OrbOut orb_vertex(uint vid [[vertex_id]], constant OrbUniforms &u [[buffer(0)]]) {
            float bound = u.centre.w * orb_bound;
            float3 centre = u.centre.xyz;
            if (u.plate.w > 0.5) {
                float4 b = orb_plate_bound(u.plate);
                centre += (u.orient * float4(b.xyz, 0.0)).xyz * u.centre.w;
                bound = b.w * u.centre.w;
            }
            float3 to = centre - u.eye.xyz;
            float d = max(length(to), bound * 1.05);
            float3 f = to / max(length(to), 1e-4);
            float3 lift = abs(f.y) > 0.95 ? float3(1.0, 0.0, 0.0) : float3(0.0, 1.0, 0.0);
            float3 r = normalize(cross(f, lift));
            float3 up = cross(r, f);
            float half_side = bound * d / sqrt(d * d - bound * bound);
            float2 corner = float2((vid & 1) ? 1.0 : -1.0, (vid & 2) ? -1.0 : 1.0);
            float3 w = centre + (r * corner.x + up * corner.y) * half_side;
            OrbOut out;
            out.position = u.viewProjection * float4(w, 1.0);
            out.world = w;
            return out;
        }

        bool orb_sphere(float3 ro, float3 rd, float r, thread float &t0, thread float &t1) {
            float b = dot(ro, rd);
            float c = dot(ro, ro) - r * r;
            float h = b * b - c;
            if (h < 0.0) return false;
            h = sqrt(h);
            t0 = -b - h;
            t1 = -b + h;
            return t1 > 0.0;
        }

        float orb_torus(float3 p, float y, float major, float minor) {
            float3 q = p - float3(0.0, y, 0.0);
            return length(float2(length(q.xz) - major, q.y)) - minor;
        }

        float orb_cylinder(float3 p, float y0, float y1, float r) {
            float2 d = float2(length(p.xz) - r, abs(p.y - (y0 + y1) * 0.5) - (y1 - y0) * 0.5);
            return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
        }

        // The cage, in the glass's radii, upright: eight iron straps along
        // the meridians over the glass, a brass collar ring top and bottom
        // with rivets round it, a stepped cap with a finial and a stepped
        // base, and a porthole facing the head where no strap crosses,
        // framed by a riveted brass bezel. material 0 is iron, 1 brass.
        float orb_cage(float3 p, float4 facing, thread int &material) {
            float a = atan2(p.z, p.x);
            float strap_step = M_PI_F / 4.0;
            float sa = a - strap_step * round(a / strap_step);
            float across = length(p.xz) * sin(abs(sa));
            float shell = abs(length(p) - 1.045) - 0.026;
            float strap = max(max(shell, across - 0.032), abs(p.y) - 0.87);
            // The porthole: the straps stop at the bezel's edge.
            float3 f = facing.xyz;
            float window = acos(clamp(facing.w, -1.0, 1.0));
            float off = acos(clamp(dot(normalize(p), f), -1.0, 1.0));
            strap = max(strap, (window + 0.05 - off) * length(p));
            float along = dot(p, f);
            float3 radial = p - f * along;
            float ring = length(radial);
            float bezel = length(float2(ring - 1.05 * sin(window), along - 1.05 * cos(window))) - 0.042;
            float3 e1 = normalize(cross(float3(0.0, 1.0, 0.0), f));
            float3 e2 = cross(f, e1);
            float ba = atan2(dot(p, e2), dot(p, e1));
            float bolt_step = M_PI_F / 5.0;
            float bb = ba - bolt_step * round(ba / bolt_step);
            float2 bq = float2(ring * cos(bb), ring * sin(bb));
            float bolt = length(float3(bq.x - 1.07 * sin(window), bq.y, along - 1.07 * cos(window))) - 0.03;
            bezel = min(bezel, bolt);
            float collar = min(orb_torus(p, 0.86, 0.56, 0.07), orb_torus(p, -0.86, 0.56, 0.07));
            float rivet_step = M_PI_F / 6.0;
            float ra = a - rivet_step * round(a / rivet_step);
            float rl = length(p.xz);
            float3 rq = float3(rl * cos(ra), abs(p.y), rl * sin(ra));
            float rivet = length(rq - float3(0.63, 0.86, 0.0)) - 0.03;
            float cap = min(orb_cylinder(p, 0.9, 1.04, 0.36) - 0.01, orb_cylinder(p, 1.04, 1.15, 0.2) - 0.01);
            cap = min(cap, length(p - float3(0.0, 1.22, 0.0)) - 0.07);
            float base = min(orb_cylinder(p, -1.04, -0.9, 0.4) - 0.01, orb_cylinder(p, -1.17, -1.04, 0.52) - 0.015);
            float brass = min(min(min(collar, rivet), min(cap, base)), bezel);
            material = strap < brass ? 0 : 1;
            return min(strap, brass);
        }

        float orb_cage_d(float3 p, float4 facing) {
            int m;
            return orb_cage(p, facing, m);
        }

        // A dim warm street at night all round: the sky dark, the ground
        // a little lit, a band of sodium light near the horizon.
        float3 orb_environment(float3 d) {
            float y = d.y;
            float3 sky = mix(float3(0.035, 0.035, 0.05), float3(0.008, 0.008, 0.014), saturate(y));
            float3 ground = float3(0.04, 0.03, 0.022);
            float3 c = y > 0.0 ? sky : mix(sky, ground, saturate(-y * 3.0));
            c += float3(0.55, 0.32, 0.12) * exp(-abs(y) * 9.0) * 0.35;
            return c;
        }

        float orb_hash(float3 p) {
            p = fract(p * 0.3183099 + 0.1);
            p *= 17.0;
            return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
        }

        float orb_noise(float3 p) {
            float3 i = floor(p), f = fract(p);
            f = f * f * (3.0 - 2.0 * f);
            float n000 = orb_hash(i), n100 = orb_hash(i + float3(1, 0, 0));
            float n010 = orb_hash(i + float3(0, 1, 0)), n110 = orb_hash(i + float3(1, 1, 0));
            float n001 = orb_hash(i + float3(0, 0, 1)), n101 = orb_hash(i + float3(1, 0, 1));
            float n011 = orb_hash(i + float3(0, 1, 1)), n111 = orb_hash(i + float3(1, 1, 1));
            return mix(mix(mix(n000, n100, f.x), mix(n010, n110, f.x), f.y),
                       mix(mix(n001, n101, f.x), mix(n011, n111, f.x), f.y), f.z);
        }

        // Iron and brass: diffuse and an anisotropic highlight brushed round
        // the vertical axis (Ward), the environment in the reflection,
        // dark patina in the crevices from the distance field's occlusion,
        // brass worn bright where noise and exposure say so.
        float3 orb_metal_lit(float3 p, float3 n, float occlusion, float3 rd, int material, float3 l, float4x4 orient);

        float3 orb_metal(float3 p, float3 rd, int material, float4 facing, float3 l, float4x4 orient) {
            const float e = 0.002;
            float3 n = normalize(float3(1, -1, -1) * orb_cage_d(p + float3(1, -1, -1) * e, facing)
                                 + float3(-1, -1, 1) * orb_cage_d(p + float3(-1, -1, 1) * e, facing)
                                 + float3(-1, 1, -1) * orb_cage_d(p + float3(-1, 1, -1) * e, facing)
                                 + float3(1, 1, 1) * orb_cage_d(p + float3(1, 1, 1) * e, facing));
            float occlusion = saturate(orb_cage_d(p + n * 0.05, facing) / 0.05) * 0.6
                + saturate(orb_cage_d(p + n * 0.12, facing) / 0.12) * 0.4;
            return orb_metal_lit(p, n, occlusion, rd, material, l, orient);
        }

        float3 orb_metal_lit(float3 p, float3 n, float occlusion, float3 rd, int material, float3 l, float4x4 orient) {
            float grime = orb_noise(p * 14.0);
            float wear = smoothstep(0.55, 0.85, grime * occlusion + 0.25 * saturate(n.y));
            float3 iron = mix(float3(0.035, 0.032, 0.03), float3(0.11, 0.1, 0.095), wear);
            float3 brass = mix(float3(0.09, 0.11, 0.07), float3(0.55, 0.38, 0.15), saturate(occlusion * 1.2) * (0.6 + 0.4 * wear));
            float3 albedo = material == 0 ? iron : brass;
            float3 f0 = material == 0 ? float3(0.35, 0.35, 0.37) * (0.5 + 0.5 * wear) : brass * 1.4;
            float ndl = saturate(dot(n, l));
            float3 h = normalize(l - rd);
            float3 t = normalize(cross(n, float3(0.0, 1.0, 0.0)) + 1e-4);
            float3 b = cross(n, t);
            float ax = 0.45, ay = 0.09;
            float th = dot(t, h) / ax, bh = dot(b, h) / ay, nh = max(dot(n, h), 1e-3);
            float ward = exp(-2.0 * (th * th + bh * bh) / (1.0 + nh)) / (4.0 * M_PI_F * ax * ay * sqrt(max(ndl * saturate(-dot(n, rd)), 1e-3)));
            float3 env = orb_environment((orient * float4(reflect(rd, n), 0.0)).xyz);
            float3 c = albedo * (ndl * 0.9 + 0.12) * occlusion;
            c += f0 * (ward * ndl * 0.35 + env * 1.2) * occlusion;
            return c;
        }

        // The length of a segment of the ray, from a to b along it, that is
        // below the liquid's surface.
        float orb_wet(float3 o, float3 d, float a, float b, float4 plane) {
            float sa = dot(o + d * a, plane.xyz) - plane.w;
            float sb = dot(o + d * b, plane.xyz) - plane.w;
            if (sa >= 0.0 && sb >= 0.0) return 0.0;
            if (sa < 0.0 && sb < 0.0) return b - a;
            float tc = a + (b - a) * sa / (sa - sb);
            return sa < 0.0 ? tc - a : b - tc;
        }

        // A print on the porthole's glass: where the glass point lies, seen
        // straight along the porthole's axis, inside the print's rectangle,
        // the texture's straight colour and alpha there.
        // Bold widens the letters by taking the most of a few samples a
        // texel or so round each.
        float4 orb_print(float3 p, float4 rect, float4 uvs, texture2d<float> tex, bool bold) {
            float2 q = (p.xy - rect.xy) / rect.zw;
            if (any(abs(q) > 1.0) || p.z <= 0.0) return float4(0.0);
            float2 uv = mix(uvs.xy, uvs.zw, float2(q.x * 0.5 + 0.5, 0.5 - q.y * 0.5));
            constexpr sampler s(filter::linear, mip_filter::linear, address::clamp_to_edge);
            float4 c = tex.sample(s, uv);
            if (bold) {
                float2 texel = 1.1 / float2(tex.get_width(), tex.get_height());
                for (int k = 0; k < 4; k++) {
                    float2 o = float2(k == 0 ? 1.0 : k == 1 ? -1.0 : 0.0, k == 2 ? 1.0 : k == 3 ? -1.0 : 0.0) * texel;
                    float4 n = tex.sample(s, uv + o);
                    if (n.a > c.a) c = n;
                }
            }
            if (c.a < 0.004) return float4(0.0);
            return float4(arm_linear(min(c.rgb / c.a, 1.0)), c.a);
        }

        // A line set on an arc: angle across it left to right, radius from
        // the letters' foot (outer) to their tops (inner), so each letter
        // stands upright to the arc. backing is the soft dark band under it.
        float4 orb_arc_print(float3 p, float4 arc, float4 uvs, texture2d<float> tex, bool bold, thread float &backing) {
            float r = length(p.xy);
            float a = atan2(p.y, p.x);
            float height = arc.y - arc.x;
            float pad = height * 0.35;
            float padAngle = pad / max(r, 1e-3);
            float radial = min(r - (arc.x - pad), (arc.y + pad) - r);
            float along = min(a - (arc.z - padAngle), (arc.w + padAngle) - a);
            float soft = pad * 0.6;
            backing = max(backing, saturate(radial / soft) * saturate(along * r / soft) * step(0.0, p.z));
            if (r < arc.x || r > arc.y || a < arc.z || a > arc.w || p.z <= 0.0) return float4(0.0);
            float2 q = float2((a - arc.z) / (arc.w - arc.z), (arc.y - r) / height);
            float2 uv = float2(mix(uvs.x, uvs.z, q.x), mix(uvs.w, uvs.y, q.y));
            constexpr sampler s(filter::linear, mip_filter::linear, address::clamp_to_edge);
            float4 c = tex.sample(s, uv);
            if (bold) {
                float2 texel = 1.1 / float2(tex.get_width(), tex.get_height());
                for (int k = 0; k < 4; k++) {
                    float2 o = float2(k == 0 ? 1.0 : k == 1 ? -1.0 : 0.0, k == 2 ? 1.0 : k == 3 ? -1.0 : 0.0) * texel;
                    float4 n = tex.sample(s, uv + o);
                    if (n.a > c.a) c = n;
                }
            }
            if (c.a < 0.004) return float4(0.0);
            return float4(arm_linear(min(c.rgb / c.a, 1.0)), c.a);
        }

        // What an orb shows at a pixel, and how far along the eye's ray (in
        // glass radii) the thing shown is: the cage, the glass, or for the
        // glow the ray's nearest pass.
        // The nameplate and its two hangers, in the orb's frame: a thin
        // brass plate with rounded edges facing the porthole's way, hung
        // from the base's foot.
        float orb_plate_d(float3 p, float4 plate) {
            float3 q = p - float3(0.0, plate.x, plate_z);
            float3 e = abs(q) - float3(plate.y - 0.03, plate.z - 0.03, 0.035 - 0.015);
            float board = length(max(e, 0.0)) + min(max(e.x, max(e.y, e.z)), 0.0) - 0.03;
            float top = plate.x + plate.z, mid = (plate_hang + top) * 0.5, half_run = (plate_hang - top) * 0.5;
            float hx = min(0.32, plate.y * 0.55);
            float3 r = float3(abs(p.x) - hx, p.y - mid, p.z - plate_z * 0.75);
            float3 re = abs(r) - float3(0.03, half_run + 0.02, 0.03);
            float hanger = length(max(re, 0.0)) + min(max(re.x, max(re.y, re.z)), 0.0);
            return min(board, hanger);
        }

        // The nameplate's colour: brass lit as the cage's, the name's letters
        // cut through its face and lit warm from behind so they read at any
        // light; the letters' dark rims sell the cut.
        float4 orb_plate_colour(float3 ro, float3 rd, constant OrbUniforms &u, texture2d<float> label, thread float &hit) {
            float4 b = orb_plate_bound(u.plate);
            float t0, t1;
            if (!orb_sphere(ro - b.xyz, rd, b.w, t0, t1)) discard_fragment();
            float t = max(t0, 0.0);
            for (int i = 0; i < 48 && t < t1; i++) {
                float3 p = ro + rd * t;
                float d = orb_plate_d(p, u.plate);
                if (d < 0.0015) {
                    const float e = 0.002;
                    float3 n = normalize(float3(1, -1, -1) * orb_plate_d(p + float3(1, -1, -1) * e, u.plate)
                                         + float3(-1, -1, 1) * orb_plate_d(p + float3(-1, -1, 1) * e, u.plate)
                                         + float3(-1, 1, -1) * orb_plate_d(p + float3(-1, 1, -1) * e, u.plate)
                                         + float3(1, 1, 1) * orb_plate_d(p + float3(1, 1, 1) * e, u.plate));
                    float3 c = orb_metal_lit(p, n, 0.85, rd, 1, u.light.xyz, u.orient);
                    if (n.z > 0.7) {
                        float2 q = float2(p.x, p.y - u.plate.x);
                        constexpr sampler s(filter::linear, mip_filter::linear, address::clamp_to_edge);
                        for (int k = 0; k < 4; k++) {
                            if (u.kinds[k] < 5.5) continue;
                            float4 at = u.at[k];
                            float2 g = (q - at.xy) / at.zw;
                            if (any(abs(g) > 1.0)) continue;
                            float2 uv = mix(u.from[k].xy, u.from[k].zw, float2(g.x * 0.5 + 0.5, 0.5 - g.y * 0.5));
                            float a = label.sample(s, uv).a;
                            float rim = label.sample(s, uv + float2(0.0, 1.5 / float(label.get_height()))).a;
                            c *= 1.0 - saturate(rim - a) * 0.6;
                            c = mix(c, float3(1.0, 0.74, 0.42) * 1.5, a);
                        }
                    }
                    hit = t;
                    return float4(c, 1.0);
                }
                t += d;
            }
            discard_fragment();
            return float4(0.0);
        }

        float4 orb_colour(OrbOut in, constant OrbUniforms &u, texture2d<float> picture, texture2d<float> label,
                          thread float &hit) {
            float radius = u.centre.w;
            if (u.plate.w > 0.5) {
                float3x3 toPlate = transpose(float3x3(u.orient[0].xyz, u.orient[1].xyz, u.orient[2].xyz));
                return orb_plate_colour(toPlate * ((u.eye.xyz - u.centre.xyz) / radius),
                                        toPlate * normalize(in.world - u.eye.xyz), u, label, hit);
            }
            float3x3 toLocal = transpose(float3x3(u.orient[0].xyz, u.orient[1].xyz, u.orient[2].xyz));
            float3 ro = toLocal * ((u.eye.xyz - u.centre.xyz) / radius);
            float3 rd = toLocal * normalize(in.world - u.eye.xyz);
            float tb0, tb1;
            if (!orb_sphere(ro, rd, orb_bound, tb0, tb1)) discard_fragment();
            float tg0 = 0.0, tg1 = 0.0;
            bool glass = orb_sphere(ro, rd, 1.0, tg0, tg1) && tg0 > 0.0;
            float3 l = u.light.xyz;
            float level = u.misc.x, time = u.misc.y;
            // The cage in front of the glass, by sphere tracing; what is
            // behind the glass is lost in the liquid and the glass's tint.
            float t = max(tb0, 0.0), stop = glass ? tg0 : tb1;
            int material = 0;
            for (int i = 0; i < 40 && t < stop; i++) {
                float d = orb_cage(ro + rd * t, u.facing, material);
                if (d < 0.0015) {
                    hit = t;
                    return float4(orb_metal(ro + rd * t, rd, material, u.facing, l, u.orient), 1.0);
                }
                t += d;
            }
            if (!glass) {
                // A faint glow of the liquid round the glass, by how near
                // the ray passes.
                float pass = length(cross(ro, rd));
                float glow = u.colour.a * (0.15 + 0.85 * level) * exp(-max(pass - 1.0, 0.0) * 14.0) * 0.22;
                hit = max(-dot(ro, rd), 0.0);
                if (glow < 0.004) discard_fragment();
                return float4(u.colour.rgb * glow, glow * 0.6);
            }
            float3 p = ro + rd * tg0;
            hit = tg0;
            float3 n = p;
            float cosi = saturate(-dot(rd, n));
            float fresnel = 0.04 + 0.96 * pow(1.0 - cosi, 5.0);
            // Through the glass: bent at the outer surface, then a thin
            // shell to the inner one, where the liquid and the icon are.
            float3 rr = refract(rd, n, 1.0 / 1.5);
            float ti0, ti1;
            orb_sphere(p, rr, 0.965, ti0, ti1);
            float a0 = max(ti0, 0.0), a1 = max(ti1, a0);
            float4 plane = u.plane;
            plane.w += 0.02 * sin(time * 1.7 + p.x * 4.0) * (level > 0.0 && level < 1.0 ? 1.0 : 0.0);
            // Beer and Lambert per channel: the liquid lets its own colour
            // through and takes the rest, so thicker reads deeper and darker.
            float3 absorb = (1.0 - u.colour.rgb) * 3.2 + 0.35;
            float3 shade = float3(0.0);
            float3 through = float3(1.0);
            float front = orb_wet(p, rr, a0, a1, plane);
            float3 glow = u.colour.rgb * (0.35 + 0.65 * saturate(dot(n, l) * 0.5 + 0.6));
            float3 seg = exp(-absorb * front);
            shade += through * glow * (1.0 - seg) * 0.55;
            through *= seg;
            // The empty part reads as glass, not a hole: a smoked frost
            // with a dim ghost of the full colour, by how much of it is
            // looked through.
            float empty = max(a1 - a0 - front, 0.0);
            float frost = 1.0 - exp(-empty * 0.9);
            shade += float3(0.08, 0.08, 0.088) * frost;
            through *= 1.0 - frost * 0.72;

            // The surface where the ray crosses it: a sheen seen from above
            // or below, and the meniscus bright where it meets the glass.
            float sa = dot(p + rr * a0, plane.xyz) - plane.w, sb = dot(p + rr * a1, plane.xyz) - plane.w;
            if (sa * sb < 0.0 && level > 0.0) {
                float tc = a0 + (a1 - a0) * sa / (sa - sb);
                float3 c = p + rr * tc;
                float wall = length(c - plane.xyz * dot(c, plane.xyz));
                float meniscus = smoothstep(0.78, 0.96, wall);
                float sheen = pow(saturate(dot(reflect(rr, plane.xyz), l)), 40.0);
                float3 surface = u.colour.rgb * (0.5 + 0.8 * meniscus) + sheen * 0.6;
                float weight = sa > 0.0 ? 0.55 : 0.3;
                shade = mix(shade, surface, weight * (0.6 + 0.4 * meniscus));
            }
            // The straps' shade on the glass under them.
            float strap = saturate(orb_cage_d(p, u.facing) / 0.06);
            shade *= 0.45 + 0.55 * strap;
            float inner = 1.0 - dot(through, float3(1.0 / 3.0));
            float3 glassTint = float3(0.03, 0.035, 0.04);
            float interiorA = saturate(inner + 0.18);
            float3 interior = shade + glassTint * (1.0 - inner);
            // The glass: the environment in its reflection, a sharp highlight
            // and a softer one, and the shell bright at the edge.
            float3 reflected = orb_environment((u.orient * float4(reflect(rd, n), 0.0)).xyz);
            float3 h = normalize(l - rd);
            float spec = pow(saturate(dot(n, h)), 600.0) * 4.0 + pow(saturate(dot(n, h)), 40.0) * 0.12;
            float rim = smoothstep(0.25, 0.0, cosi) * 0.22;
            float3 colour = interior * (1.0 - fresnel) + (reflected + rim) * fresnel;
            float alpha = saturate(interiorA * (1.0 - fresnel) + fresnel + rim);
            // The level's ring where the surface meets the glass, glowing,
            // and fine ticks engraved down the porthole's outer edge at a
            // quarter, a half and three quarters full, the half longest.
            if (level > 0.0 && level < 1.0) {
                float d = dot(n * 0.965, plane.xyz) - plane.w;
                // A thin bright line only: a wide coloured band read as a
                // film of blood left on the glass (headset run, 2026-11-25).
                float ring = exp(-abs(d) * 160.0);
                colour += (u.colour.rgb * 1.6 + float3(0.55)) * ring;
                alpha = saturate(alpha + ring * 0.8);
            }
            {
                float az = atan2(n.x, n.z) * u.ticks.w;
                float h = dot(n, plane.xyz);
                float tick = 0.0;
                for (int k = 0; k < 3; k++) {
                    float reach = k == 1 ? 0.12 : 0.07;
                    float px = fwidth(h) * 1.2 + 0.004;
                    tick = max(tick, (1.0 - smoothstep(0.0, px, abs(h - u.ticks[k]))) * step(abs(az - 0.82), reach));
                }
                float spine = (1.0 - smoothstep(0.0, fwidth(az) * 1.2 + 0.004, abs(az - (0.82 + 0.07)))) * step(abs(h - u.ticks[1]), 0.62);
                float etch = max(tick, spine * 0.6) * step(0.0, n.z);
                colour = colour * (1.0 - etch * 0.7) + float3(0.85, 0.86, 0.9) * etch * 0.7;
                alpha = saturate(alpha + etch * 0.7);
            }
            // The prints on the porthole's glass, lit from behind so they
            // read over the liquid at any level: a smoked backing under
            // each, then the print itself, and the glass's highlight and
            // reflection passing over both.
            if (dot(n, u.facing.xyz) > u.facing.w) {
                // The picture first, then the bands under the lettering,
                // then the lettering.
                float backing = 0.0;
                float4 prints[4];
                for (int k = 0; k < 4; k++) {
                    float kind = u.kinds[k];
                    prints[k] = kind < 0.5 ? float4(0.0)
                        : kind < 1.5 ? orb_print(n, u.at[k], u.from[k], picture, false)
                                     : orb_arc_print(n, u.at[k], u.from[k], label, kind > 4.5, backing);
                }
                for (int k = 0; k < 4; k++) {
                    if (u.kinds[k] < 0.5 || u.kinds[k] > 1.5) continue;
                    float4 pr = prints[k];
                    if (pr.a <= 0.0) continue;
                    colour = colour * (1.0 - pr.a * 0.55);
                    alpha = alpha + (1.0 - alpha) * pr.a * 0.55;
                    colour = colour * (1.0 - pr.a) + (pr.rgb * 1.25 + 0.04) * pr.a;
                    alpha = alpha + (1.0 - alpha) * pr.a;
                }
                colour *= 1.0 - backing * 0.7;
                alpha = alpha + (1.0 - alpha) * backing * 0.7;
                for (int k = 0; k < 4; k++) {
                    if (u.kinds[k] < 3.5) continue;
                    float4 pr = prints[k];
                    if (pr.a <= 0.0) continue;
                    float3 lit = pr.rgb * 1.25 + 0.04;
                    colour = colour * (1.0 - pr.a) + lit * pr.a;
                    alpha = alpha + (1.0 - alpha) * pr.a;
                }
                colour += reflected * fresnel * 0.5;
            }
            colour += spec * strap;
            alpha = saturate(alpha + spec);
            return float4(colour, alpha);
        }

        fragment float4 orb_fragment(OrbOut in [[stage_in]], constant OrbUniforms &u [[buffer(0)]],
                                     texture2d<float> picture [[texture(0)]], texture2d<float> label [[texture(1)]]) {
            float hit = 0.0;
            return orb_colour(in, u, picture, label, hit);
        }

        struct OrbDepthOut {
            float4 colour [[color(0)]];
            float depth [[depth(any)]];
        };

        // With our hands drawn: the depth of what the pixel shows, for the
        // test against theirs.
        fragment OrbDepthOut orb_fragment_depth(OrbOut in [[stage_in]], constant OrbUniforms &u [[buffer(0)]],
                                                texture2d<float> picture [[texture(0)]],
                                                texture2d<float> label [[texture(1)]]) {
            float hit = 0.0;
            OrbDepthOut out;
            out.colour = orb_colour(in, u, picture, label, hit);
            float3 dir = normalize(in.world - u.eye.xyz);
            float4 c = u.viewProjection * float4(u.eye.xyz + dir * (hit * u.centre.w), 1.0);
            out.depth = saturate(c.z / c.w);
            return out;
        }

        struct IconOut {
            float4 position [[position]];
            float2 uv;
        };

        vertex IconOut icon_vertex(uint vid [[vertex_id]]) {
            float2 uv = float2((vid << 1) & 2, vid & 2);
            IconOut out;
            out.position = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
            out.uv = uv;
            return out;
        }

        // The game's icon at its own pixels, premultiplied: its alpha is
        // straight, and a filter must work on premultiplied colour or the
        // clear edges bleed dark.
        fragment float4 icon_decode(IconOut in [[stage_in]], texture2d<float> src [[texture(0)]]) {
            constexpr sampler s(filter::nearest, address::clamp_to_edge);
            float4 c = src.sample(s, in.uv);
            return float4(c.rgb * c.a, c.a);
        }

        float lanczos2(float x) {
            x = abs(x);
            if (x < 1e-4) return 1.0;
            if (x >= 2.0) return 0.0;
            float px = M_PI_F * x;
            return 2.0 * sin(px) * sin(px * 0.5) / (px * px);
        }

        // Lanczos with two lobes over the 4x4 texels round each output
        // pixel, then a light unsharp mask against the bilinear sample.
        fragment float4 icon_lanczos(IconOut in [[stage_in]], texture2d<float> src [[texture(0)]]) {
            constexpr sampler s(filter::linear, address::clamp_to_edge);
            int2 size = int2(src.get_width(), src.get_height());
            float2 pos = in.uv * float2(size) - 0.5;
            float2 base = floor(pos);
            float2 f = pos - base;
            float4 sum = 0.0;
            float weights = 0.0;
            for (int j = -1; j <= 2; j++) {
                for (int i = -1; i <= 2; i++) {
                    float w = lanczos2(float(i) - f.x) * lanczos2(float(j) - f.y);
                    int2 t = clamp(int2(base) + int2(i, j), int2(0), size - 1);
                    sum += src.read(uint2(t)) * w;
                    weights += w;
                }
            }
            float4 c = sum / max(weights, 1e-4);
            c += 0.3 * (c - src.sample(s, in.uv));
            c = clamp(c, 0.0, 1.0);
            return float4(min(c.rgb, c.a), c.a);
        }

        // A rectangle of a texture's alpha, from origin, at its pixels.
        fragment float alpha_mask(IconOut in [[stage_in]], texture2d<float> src [[texture(0)]],
                                  constant uint2 &origin [[buffer(0)]]) {
            return src.read(origin + uint2(in.position.xy)).a;
        }

        // MetalFX's colour with the Lanczos alpha, kept premultiplied.
        fragment float4 icon_merge(IconOut in [[stage_in]], texture2d<float> fx [[texture(0)]],
                                   texture2d<float> lz [[texture(1)]]) {
            uint2 at = uint2(in.position.xy);
            float a = lz.read(at).a;
            return float4(min(fx.read(at).rgb, a), a);
        }
        """
}

/// The late rotation's size, logged every five seconds: how far the head
/// had turned between the pose a frame was drawn for and the compositor
/// frame showing it, which is what the head-locked plane used to drag the
/// world by. Touched by the compositor's thread only.
private struct LateRotationStats {
    static let period: Double = 5

    var start = CACurrentMediaTime()
    var turned = 0
    var unturned = 0
    var sum: Double = 0
    var max: Float = 0
    var over1 = 0
    var over5 = 0
    var ageSum: Double = 0

    mutating func add(degrees: Float, age: Double) {
        turned += 1
        sum += Double(degrees)
        if degrees > max { max = degrees }
        if degrees > 1 { over1 += 1 }
        if degrees > 5 { over5 += 1 }
        ageSum += age
    }

    mutating func addUnturned() {
        unturned += 1
    }

    mutating func logIfDue() {
        let now = CACurrentMediaTime()
        guard now - start >= Self.period else { return }
        if turned > 0 || unturned > 0 {
            let mean = turned > 0 ? sum / Double(turned) : 0
            let age = turned > 0 ? ageSum / Double(turned) : 0
            ScreenCompositor.log(
                String(
                    format: "late rotation over %.1f s: %d stereo frames turned, mean %.2f deg, max %.2f deg, "
                        + "%d over 1 deg, %d over 5 deg, frame age mean %.1f ms; %d left head locked (no pose)",
                    now - start, turned, mean, Double(max), over1, over5, age, unturned
                )
            )
        }
        self = LateRotationStats(start: now)
    }
}

/// The reticle's constants, laid out as the shader's.
private struct ReticleUniforms {
    var mvp: simd_float4x4
    var colour: SIMD4<Float>
    var style: Int32
    /// The gauges' fill, 0 to 1, or -1 for none.
    var health: Float = -1
    var blood: Float = -1
    /// Half the quad's side in the shape's units: 1 for the reticle
    /// alone, more to fit a wider ring and the gauges round it.
    var extent: Float = 1
    /// The ring's radius in the shape's units, which follows the spread,
    /// and the gauges' radius outside it.
    var ring: Float = 0.7
    var gaugeRadius: Float = 1.12
}

/// An orb's constants, laid out as the shader's.
private struct OrbUniforms {
    var viewProjection: simd_float4x4
    var centre: SIMD4<Float>
    var eye: SIMD4<Float>
    var colour: SIMD4<Float>
    var plane: SIMD4<Float>
    var misc: SIMD4<Float>
    var facing: SIMD4<Float>
    var at0: SIMD4<Float>
    var at1: SIMD4<Float>
    var at2: SIMD4<Float>
    var at3: SIMD4<Float>
    var from0: SIMD4<Float>
    var from1: SIMD4<Float>
    var from2: SIMD4<Float>
    var from3: SIMD4<Float>
    var kinds: SIMD4<Float>
    var orient: simd_float4x4
    var light: SIMD4<Float>
    var ticks: SIMD4<Float>
    var plate: SIMD4<Float>
}

/// The arm HUD card's constants, laid out as the shader's: uv is the
/// side's part of the cards' texture, u0 v0 u1 v1 from the top left.
private struct ArmUniforms {
    var mvp: simd_float4x4
    var uv: SIMD4<Float>
}

/// The hand ray's constants, laid out as the shader's.
private struct HandUniforms {
    var mvp: simd_float4x4
    var colour: SIMD4<Float>
}

/// The vertex and fragment stages' constants, laid out as the shader's.
private struct ScreenUniforms {
    var mvp: simd_float4x4
    var cursor: SIMD4<Float>
}

/// The eye plane stages' constants, laid out as the shader's: 96 bytes,
/// so the padding is three scalars, not a SIMD3, whose 16-byte size would
/// make the stride 112.
private struct StereoUniforms {
    var mvp: simd_float4x4
    var cursor: SIMD4<Float>
    var slice: UInt32
    var edge: Float = 0
    var dim: Float = 0
    var pad: Float = 0
}

extension Duration {
    /// Whole and fractional seconds, as ARKit's timestamps take them.
    fileprivate var seconds: TimeInterval {
        let parts = components
        return TimeInterval(parts.seconds) + TimeInterval(parts.attoseconds) * 1e-18
    }
}
