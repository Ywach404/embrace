import ARKit
import CompositorServices
import Metal
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
        blend.isBlendingEnabled = false
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
        let texture = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm_srgb, width: 1, height: 1, mipmapped: false
        )
        guard let placeholder = device.makeTexture(descriptor: texture) else { throw CompositorError.noDevice }
        // Dark grey until the first game frame, so the screen's placement
        // shows at once.
        var grey: UInt32 = 0xFF18_1818
        placeholder.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: &grey, bytesPerRow: 4)
        self.placeholder = placeholder
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
            let on = SpatialInput.armHUD && hand.valid && hand.state >= 2
            metal_d3d9_fullview_arm_hud(Int32(hand.side.rawValue), on ? 1 : 0)
        }
        for drawable in drawables {
            let time = LayerRenderer.Clock.Instant.epoch.duration(to: drawable.frameTiming.presentationTime)
            let anchor = worldTracking.queryDeviceAnchor(atTimestamp: time.seconds)
            drawable.deviceAnchor = anchor
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
            if stereo, shown.fullview != 0, source != nil, SpatialInput.armHUD {
                encodeArmHUD(drawable: drawable, anchor: anchor, hands: hands, shown: shown, into: commandBuffer)
            }
            // The rays only while a menu's cursor is up (the inventory, the
            // sheet), where they point; in play they would be clutter.
            if SpatialInput.rayShown || cursor.z > 0 {
                encodeHands(drawable: drawable, anchor: anchor, hands: hands, into: commandBuffer)
            } else if SpatialInput.handAim, stereo, shown.fullview != 0 {
                encodeReticle(drawable: drawable, anchor: anchor, hands: hands, shown: shown, into: commandBuffer)
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

    /// Where the gun hand's shots go, in play: a mark 8 m out along
    /// its aim turned down by the hand aim tilt, as the runtime turns the
    /// shot, so the reticle is at the hand rather than at the middle of the
    /// head's view (the game's own is turned off for full view). Drawn over
    /// the frame with no depth test, like the rays.
    private func encodeReticle(
        drawable: LayerRenderer.Drawable,
        anchor: DeviceAnchor?,
        hands: [HandPose],
        shown: metal_d3d9_shown_frame,
        into commandBuffer: MTLCommandBuffer
    ) {
        guard let anchor, anchor.isTracked else { return }
        let distance: Float = 8
        let size = distance * 0.022
        let head = anchor.originFromAnchorTransform.columns.3
        let centre: SIMD3<Float>
        if rt_host_third_person() != 0 {
            // Behind the player the game aims through the middle of its own
            // frame, which is the head pose the frame was drawn for (held
            // still in third person): the mark goes there, in place of the
            // game's crosshair, which full view turns off.
            guard shown.pose_valid != 0 else { return }
            let yaw = shown.pose_yaw_deg * .pi / 180
            let pitch = shown.pose_pitch_deg * .pi / 180
            let dir = SIMD3(-sin(yaw) * cos(pitch), sin(pitch), -cos(yaw) * cos(pitch))
            centre = SIMD3(head.x, head.y, head.z) + dir * distance
        } else {
            let hand = hands[SpatialInput.aimSide.rawValue]
            guard hand.valid, hand.state >= 2 else { return }
            let tilt = -SpatialInput.aimTilt * .pi / 180
            let (c, sn) = (cos(tilt), sin(tilt))
            var turn = matrix_identity_float4x4
            turn.columns.1 = [0, c, sn, 0]
            turn.columns.2 = [0, -sn, c, 0]
            // Along the hand's aim turned down as the shot is.
            let at = hand.originFromAim * turn * SIMD4<Float>(0, 0, -distance, 1)
            centre = SIMD3(at.x, at.y, at.z)
        }
        // The square turned to face the head, so the shape reads the same
        // however the hand rolls.
        let back = simd_normalize(SIMD3(head.x, head.y, head.z) - centre)
        var right = simd_cross(SIMD3<Float>(0, 1, 0), back)
        right = simd_length(right) > 1e-4 ? simd_normalize(right) : SIMD3(1, 0, 0)
        let up = simd_cross(back, right)
        let model = simd_float4x4(
            SIMD4(right * size, 0), SIMD4(up * size, 0), SIMD4(back, 0), SIMD4(centre, 1)
        )
        let gauges = Self.reticleGauges()
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
            var uniforms = ReticleUniforms(
                mvp: projection * viewFromOrigin * model, colour: SpatialInput.reticleColour,
                style: SpatialInput.reticleStyle, health: gauges.health, blood: gauges.blood,
                extent: gauges.health >= 0 || gauges.blood >= 0 ? 1.25 : 1
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

        // One gauge: an arc of the ring at radius 1.12 (1.6 times the
        // reticle's), 120 degrees about angle 0 of the side's own angle,
        // filled from its lower end up to fraction f and dim beyond. Edges
        // are smoothed over a pixel as the reticle's are, the ends and the
        // fill's edge measured along the arc. Premultiplied out.
        float4 reticle_gauge(float2 p, float angle, float f, float3 colour, float px, float line) {
            if (f < 0.0) return float4(0.0);
            float radius = 1.12;
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
            float ring = 1.0 - smoothstep(line - px, line + px, abs(r - 0.7));
            float dot = 1.0 - smoothstep(0.14 - px, 0.14 + px, r);
            float arms = max(1.0 - smoothstep(line - px, line + px, abs(p.x)),
                             1.0 - smoothstep(line - px, line + px, abs(p.y)));
            float span = smoothstep(0.28 - px, 0.28 + px, r) * (1.0 - smoothstep(0.95 - px, 0.95 + px, r));
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
            float4 health = reticle_gauge(p, atan2(p.y, -p.x), u.health, float3(0.3, 0.72, 0.95), px, line);
            float4 blood = reticle_gauge(p, atan2(p.y, p.x), u.blood, float3(0.72, 0.04, 0.07), px, line);
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
    /// alone, more to fit the gauges round it.
    var extent: Float = 1
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
