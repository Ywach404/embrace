import ARKit
import Foundation
import GameController
import simd

/// One side's tracked pose as the compositor read it this frame: a
/// controller now, a bare hand later through the same struct.
struct HandPose {
    enum Source { case controller, hand }
    enum Side: Int { case left = 0, right = 1 }

    var valid = false
    var source = Source.controller
    var side = Side.right
    /// 0 untracked, 1 orientation only, 2 position and orientation, 3 that
    /// at low accuracy, as the runtime and backend take it.
    var state: Int32 = 0
    var originFromGrip = matrix_identity_float4x4
    var originFromAim = matrix_identity_float4x4
    var velocity = SIMD3<Float>.zero
    var angularVelocity = SIMD3<Float>.zero
    /// Seconds on ARKit's clock.
    var timestamp: TimeInterval = 0
}

/// Tracks the spatial controllers with ARKit's accessory tracking and
/// hands each side's pose on: to the runtime (`rt_host_hand_pose`) and the
/// backend (`metal_d3d9_immersive_hand_pose`), and to the compositor for
/// the debug ray.
///
/// ARKit loads an accessory from a GameController device. Whether the
/// merged Sense pair visionOS presents as one MFi gamepad is such an
/// accessory is not documented, so every controller and spatial accessory
/// that connects is tried, and the result logged with its name, its
/// chirality and its locations, or the error. The provider can only be
/// made with a list, so it is made on the first accessory that loads and
/// given the list again (`updateAccessories`) as devices come and go.
///
/// The anchors are not safe to read from two threads, so `frame(at:)` is
/// called on the compositor's thread only, once a compositor frame. The
/// accessory list and the provider are set from tasks, under the lock.
final class SpatialInput: @unchecked Sendable {
    nonisolated static let shared = SpatialInput()

    /// The box and ray in play too, for checking the tracking against the
    /// real controllers: `RT_HANDRAY=1` or `--handray`. Without it they
    /// show only while a menu's cursor is up, where they point.
    /// Hand aim, as the runtime decides it: on in full view unless
    /// `RT_HANDAIM=0`. The reticle follows the gun hand when it is.
    static let handAim: Bool = ProcessInfo.processInfo.environment["RT_HANDAIM"] != "0"

    /// The runtime's tilt of the aim, `RT_HANDAIM_TILT` degrees down, 25 if
    /// not set, so the reticle sits where the shot goes.
    static let aimTilt: Float = {
        if let v = ProcessInfo.processInfo.environment["RT_HANDAIM_TILT"], let f = Float(v), f >= -60, f <= 60 {
            return f
        }
        return 25
    }()

    /// The runtime's rest offset for the gun, `RT_HANDAIM_REST`
    /// "forward,right,up" in game units, 0,0,-3 if not set, and the game
    /// units a metre (`RT_STEREO_UNITS_PER_M`, the inch): the gun's frame
    /// the gloves are placed in is worked out as handaim.c works it.
    static let handAimRest: SIMD3<Float> = {
        let v = (ProcessInfo.processInfo.environment["RT_HANDAIM_REST"] ?? "").split(separator: ",").compactMap {
            Float($0.trimmingCharacters(in: .whitespaces))
        }
        return v.count == 3 ? SIMD3(v[0], v[1], v[2]) : SIMD3(0, 0, -3)
    }()
    static let unitsPerMetre: Float = {
        if let v = ProcessInfo.processInfo.environment["RT_STEREO_UNITS_PER_M"], let f = Float(v), f > 0 { return f }
        return Float(1.0 / 0.0254)
    }()

    /// The reticle's shape, `RT_RETICLE_STYLE`: 0 cross, 1 circle and dot,
    /// 2 dot, 3 circle, 4 cross and dot.
    static let reticleStyle: Int32 = {
        if let v = ProcessInfo.processInfo.environment["RT_RETICLE_STYLE"], let n = Int32(v), (0...4).contains(n) {
            return n
        }
        return 1
    }()

    /// Health and blood gauges curved round the reticle, health on the left
    /// and blood on the right, each filling from below: on unless
    /// `RT_RETICLE_GAUGES=0`. The game's own HUD is flat and full view
    /// leaves it out of sight, so they are drawn where the eye already is.
    static let reticleGauges: Bool = ProcessInfo.processInfo.environment["RT_RETICLE_GAUGES"] != "0"

    /// The arm HUD's old cards, in full view: the game's left gauge
    /// (health and the portrait) on a card at the left controller and its
    /// right one (blood and the discipline) at the right. Only with
    /// `RT_ARM_HUD=1`, which the backend reads too; the orbs replaced them.
    static let armHUD: Bool = ProcessInfo.processInfo.environment["RT_ARM_HUD"] == "1"

    /// The orbs, in full view: health in a blue orb beside the left
    /// controller and blood in a red one beside the right, the selected
    /// discipline's icon above it, the game's own gauges off the panel.
    /// On unless `RT_ORB_HUD=0` or the cards are asked for.
    static let orbHUD: Bool = !armHUD && ProcessInfo.processInfo.environment["RT_ORB_HUD"] != "0"

    /// Each orb's diameter in metres: `RT_ORB_SIZE` in centimetres, 3 to
    /// 30, default 8, about a fist at arm's length.
    static let orbSize: Float = {
        if let v = ProcessInfo.processInfo.environment["RT_ORB_SIZE"], let f = Float(v), f >= 3, f <= 30 {
            return f / 100
        }
        return 0.08
    }()

    /// In third person the orbs leave the hands for places beside the
    /// body, following its facing: `RT_ORB_TP_HEIGHT` centimetres below
    /// the eyes (default 15), `RT_ORB_TP_SIDE` out to each side (default
    /// 25) and `RT_ORB_TP_FORWARD` ahead (default 45), about 30 degrees
    /// off centre and a little low, where both are seen at once.
    static let orbThirdPersonHeight: Float = orbCentimetres("RT_ORB_TP_HEIGHT", default: 15, range: -50...100)
    static let orbThirdPersonSide: Float = orbCentimetres("RT_ORB_TP_SIDE", default: 25, range: 0...150)
    static let orbThirdPersonForward: Float = orbCentimetres("RT_ORB_TP_FORWARD", default: 45, range: 10...200)

    /// Which hand holds the health orb (the weapon and its ammunition):
    /// `RT_ORB_HEALTH_SIDE=right` (the default, the gun hand) or `left`;
    /// the blood orb takes the other. 1 right, 0 left, as `HandPose.Side`.
    static let orbHealthHand: Int = ProcessInfo.processInfo.environment["RT_ORB_HEALTH_SIDE"] == "left" ? 0 : 1

    /// Each orb's turn from upright with its porthole level towards the
    /// head, in degrees: pitch turns the porthole up towards the eyes (the
    /// cage's top leaning back, away from them; negative leans the top
    /// towards them, the porthole down), default 40; yaw
    /// turns it outwards (away from the body), roll turns it clockwise as
    /// seen. `RT_ORB_PITCH`, `RT_ORB_YAW`, `RT_ORB_ROLL` give both orbs
    /// the same, mirrored, so they tilt symmetrically; `RT_ORB_TILT_SPLIT`
    /// as six numbers, left hand's pitch, yaw, roll then the right's,
    /// gives each its own in the same sense (yaw outwards and roll
    /// mirrored for the left). Index 0 the left hand, 1 the right.
    static let orbTilt: [SIMD3<Float>] = {
        let env = ProcessInfo.processInfo.environment
        func degrees(_ name: String, default value: Float = 0) -> Float {
            if let v = env[name], let f = Float(v), abs(f) <= 180 { return f }
            return value
        }
        if let split = env["RT_ORB_TILT_SPLIT"] {
            let values = split.split(separator: ",").compactMap { Float($0.trimmingCharacters(in: .whitespaces)) }
            if values.count == 6, values.allSatisfy({ abs($0) <= 180 }) {
                return [SIMD3(values[0], values[1], values[2]), SIMD3(values[3], values[4], values[5])]
            }
        }
        // Pitched 40 degrees by default: the orbs sit below the eyes, so the
        // porthole is turned up to face them.
        let both = SIMD3<Float>(degrees("RT_ORB_PITCH", default: 40), degrees("RT_ORB_YAW"), degrees("RT_ORB_ROLL"))
        return [both, both]
    }()

    private static func orbCentimetres(_ name: String, default value: Float, range: ClosedRange<Float>) -> Float {
        if let v = ProcessInfo.processInfo.environment[name], let f = Float(v), range.contains(f) { return f / 100 }
        return value / 100
    }

    /// How far each orb's centre sits out from its grip, away from the
    /// body, in metres: `RT_ORB_OFFSET` in centimetres, 0 to 40, default 9,
    /// clear of the hand and the controller's ring.
    static let orbOffset: Float = {
        if let v = ProcessInfo.processInfo.environment["RT_ORB_OFFSET"], let f = Float(v), f >= 0, f <= 40 {
            return f / 100
        }
        return 0.09
    }()

    /// Each card's height in metres: `RT_ARM_HUD_SIZE` in centimetres, 5 to
    /// 60, default 28. A gauge is a tall thin column, so at 14 cm it was a
    /// sliver too narrow to read at arm's length (headset run, 2026-11-21).
    static let armHUDSize: Float = {
        if let v = ProcessInfo.processInfo.environment["RT_ARM_HUD_SIZE"], let f = Float(v), f >= 5, f <= 60 {
            return f / 100
        }
        return 0.28
    }()

    /// How far above the grip each card's lower edge sits, in metres:
    /// `RT_ARM_HUD_OFFSET` in centimetres, 0 to 30, default 10, clear of the
    /// hand holding the controller.
    static let armHUDOffset: Float = {
        if let v = ProcessInfo.processInfo.environment["RT_ARM_HUD_OFFSET"], let f = Float(v), f >= 0, f <= 30 {
            return f / 100
        }
        return 0.10
    }()

    /// How far each card leans back from upright, its face turned from
    /// the player's way up towards the sky, in degrees: `RT_ARM_HUD_TILT`,
    /// -90 to 90, default 45, so a glance down at the hand reads it.
    static let armHUDTilt: Float = {
        if let v = ProcessInfo.processInfo.environment["RT_ARM_HUD_TILT"], let f = Float(v), f >= -90, f <= 90 {
            return f
        }
        return 45
    }()

    /// Our own hands at the controllers (`Hands/`), in place of the game's
    /// first person arms, which the backend then leaves out (draw.m,
    /// hide_arms): only with `RT_HANDS=1` until they have been seen on the
    /// headset. Drawn in first person and in menus; in third person too
    /// with `RT_HANDS_TP=1`.
    static let hands: Bool = ProcessInfo.processInfo.environment["RT_HANDS"] == "1"
    static let handsInThirdPerson: Bool = ProcessInfo.processInfo.environment["RT_HANDS_TP"] == "1"

    /// The hand model's place in its controller's grip space, for fitting
    /// it to the controller: `RT_HANDS_OFFSET` as x, y, z centimetres
    /// (right, up, back in the grip's axes) and `RT_HANDS_ROT` as pitch,
    /// yaw, roll degrees, both for the right hand and mirrored for the
    /// left. The defaults put the palm round the grip with the fingers
    /// along its forward axis.
    static let handsOffset: SIMD3<Float> = {
        if let v = ProcessInfo.processInfo.environment["RT_HANDS_OFFSET"] {
            let f = v.split(separator: ",").compactMap { Float($0.trimmingCharacters(in: .whitespaces)) }
            if f.count == 3, f.allSatisfy({ abs($0) <= 50 }) { return SIMD3(f[0], f[1], f[2]) / 100 }
        }
        return SIMD3(0.03, -0.005, 0.06)
    }()
    static let handsRotation: SIMD3<Float> = {
        if let v = ProcessInfo.processInfo.environment["RT_HANDS_ROT"] {
            let f = v.split(separator: ",").compactMap { Float($0.trimmingCharacters(in: .whitespaces)) }
            if f.count == 3, f.allSatisfy({ abs($0) <= 180 }) { return SIMD3(f[0], f[1], f[2]) }
        }
        return SIMD3(-15, 0, 0)
    }()
    /// The model's scale, `RT_HANDS_SCALE`, 0.5 to 2, default 1.15: the
    /// glove as modelled read small against the player's own hands on the
    /// headset (2026-11-24). The layout edit changes it live.
    static let handsScale: Float = {
        if let v = ProcessInfo.processInfo.environment["RT_HANDS_SCALE"], let f = Float(v), f >= 0.5, f <= 2 { return f }
        return 1.15
    }()

    /// Its colour, white and nearly opaque: it is drawn over the world with
    /// no depth test, so it reads on dark streets and lit signs alike.
    static let reticleColour: SIMD4<Float> = [1, 1, 1, 0.9]

    static let rayShown: Bool = {
        if CommandLine.arguments.contains("--handray") { return true }
        return ProcessInfo.processInfo.environment["RT_HANDRAY"] == "1"
    }()

    private let lock = NSLock()
    private var session: ARKitSession?
    private var provider: AccessoryTrackingProvider?
    private var running = false
    /// Loaded accessories by the device they came from.
    private var accessories: [ObjectIdentifier: Accessory] = [:]
    /// The provider is being started, and the list changed meanwhile: one
    /// start only, then the list it missed. Two halves load at once, and
    /// running a provider already running stops the app (exit 6 on a
    /// headset run, 2026-11-21).
    private var starting = false
    private var missed = false
    private var observers: [NSObjectProtocol] = []
    /// Devices already offered to ARKit (main thread), so a repeated connect
    /// or the periodic sweep does not load one twice, and a sweep timer.
    private var offered: Set<ObjectIdentifier> = []
    private var sweep: Timer?
    private var started = false
    /// Read on the compositor's thread only.
    private var lastState: [Int32] = [-1, -1]
    private var lastSide: [UUID: HandPose.Side] = [:]
    private var frames: UInt64 = 0

    /// Starts watching for devices and asks for accessory tracking. Called
    /// when the immersive space's compositor starts; idempotent.
    func start() {
        lock.lock()
        let first = !started
        started = true
        lock.unlock()
        guard first else { return }
        guard AccessoryTrackingProvider.isSupported else {
            Self.log("accessory tracking is not supported here")
            return
        }
        let session = ARKitSession()
        lock.lock()
        self.session = session
        lock.unlock()
        Task.detached {
            let result = await session.requestAuthorization(for: [.accessoryTracking])
            let status = result[.accessoryTracking].map { "\($0)" } ?? "no answer"
            Self.log("accessory tracking authorization: \(status)")
        }
        DispatchQueue.main.async { [self] in
            let center = NotificationCenter.default
            for name in [NSNotification.Name.GCControllerDidConnect, .GCSpatialAccessoryDidConnect] {
                observers.append(
                    center.addObserver(forName: name, object: nil, queue: .main) { [self] note in
                        if let device = note.object as? GCDevice { self.load(device) }
                    }
                )
            }
            for name in [NSNotification.Name.GCControllerDidDisconnect, .GCSpatialAccessoryDidDisconnect] {
                observers.append(
                    center.addObserver(forName: name, object: nil, queue: .main) { [self] note in
                        if let device = note.object as? GCDevice { self.drop(device) }
                    }
                )
            }
            let controllers = GCController.controllers()
            let spatial = GCSpatialAccessory.spatialAccessories
            Self.log("tracking probe at start: \(controllers.count) controllers, \(spatial.count) spatial accessories")
            for device in controllers as [GCDevice] + spatial as [GCDevice] { load(device) }
            // A controller that comes on late is not always announced: the
            // list itself is read again each second, and what is new is
            // loaded, what is gone dropped.
            sweep = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [self] _ in
                MainActor.assumeIsolated { self.resweep() }
            }
        }
    }

    /// Stops the session when the space closes. A later space starts again.
    func stop() {
        lock.lock()
        let session = session
        self.session = nil
        provider = nil
        running = false
        accessories = [:]
        started = false
        lock.unlock()
        session?.stop()
        DispatchQueue.main.async { [self] in
            for o in observers { NotificationCenter.default.removeObserver(o) }
            observers = []
            sweep?.invalidate()
            sweep = nil
            offered = []
        }
    }

    /// The devices there now, controllers and spatial accessories.
    private static func present() -> [GCDevice] {
        GCController.controllers() as [GCDevice] + GCSpatialAccessory.spatialAccessories as [GCDevice]
    }

    private func resweep() {
        let now = Self.present()
        let ids = Set(now.map { ObjectIdentifier($0 as AnyObject) })
        for device in now where !offered.contains(ObjectIdentifier(device as AnyObject)) { load(device) }
        let gone = offered.subtracting(ids)
        guard !gone.isEmpty else { return }
        offered.subtract(gone)
        let had = lock.withLock { gone.filter { accessories.removeValue(forKey: $0) != nil }.count }
        if had > 0 {
            Self.log("accessory gone: \(had) no longer listed")
            Task.detached { [self] in await apply() }
        }
    }

    private func load(_ device: GCDevice) {
        let label = Self.label(device)
        let id = ObjectIdentifier(device as AnyObject)
        // Once per device: the pad layer's re-announcement of a controller
        // that changed in place (ControllerWatcher) posts a connect for one
        // already here.
        guard offered.insert(id).inserted else { return }
        // GCDevice is not Sendable; the framework objects are safe to hand
        // to ARKit's loader from another task.
        let box = DeviceBox(device: device)
        Task.detached { [self] in
            do {
                let accessory = try await Accessory(device: box.device)
                let locations = accessory.locations.map(\.rawValue).joined(separator: ", ")
                Self.log(
                    "accessory loaded from \(label): \"\(accessory.name)\", inherent \(accessory.inherentChirality), "
                        + "locations [\(locations)]"
                )
                lock.withLock { accessories[id] = accessory }
                await apply()
            } catch {
                Self.log("no accessory from \(label): \(error)")
            }
        }
    }

    private func drop(_ device: GCDevice) {
        let id = ObjectIdentifier(device as AnyObject)
        // A disconnect posted for a controller still listed is the pad
        // layer's re-announcement, not a real loss: tracking keeps it.
        if Self.present().contains(where: { $0 === device }) { return }
        offered.remove(id)
        lock.lock()
        let had = accessories.removeValue(forKey: id) != nil
        lock.unlock()
        guard had else { return }
        Self.log("accessory gone with \(Self.label(device))")
        Task.detached { [self] in await apply() }
    }

    /// Makes and runs the provider on the first accessory, and gives it the
    /// whole list after that.
    private func apply() async {
        let (list, session, provider, isRunning, busy) = lock.withLock {
            () -> ([Accessory], ARKitSession?, AccessoryTrackingProvider?, Bool, Bool) in
            let list = Array(accessories.values)
            if self.provider == nil, !list.isEmpty {
                self.provider = AccessoryTrackingProvider(accessories: list)
            }
            let busy = starting
            if busy { missed = true } else if !running, self.provider != nil { starting = true }
            return (list, self.session, self.provider, running, busy)
        }
        guard !busy, let session, let provider else { return }
        do {
            if !isRunning {
                do {
                    try await session.run([provider])
                } catch {
                    lock.withLock { starting = false }
                    throw error
                }
                let again = lock.withLock { () -> Bool in
                    running = true
                    starting = false
                    defer { missed = false }
                    return missed
                }
                Self.log("accessory tracking running with \(list.count) accessories")
                if again { await apply() }
            } else {
                try await provider.updateAccessories(list)
                Self.log("accessory tracking updated: \(list.count) accessories")
            }
        } catch {
            Self.log("accessory tracking did not take the list: \(error)")
        }
    }

    /// Both sides' poses, predicted to `time` (the drawable's trackable
    /// anchor time) with rendering correction, so what the compositor draws
    /// sits on the real controller. Publishes the unpredicted latest to the
    /// runtime and the backend, since game aim wants precision over a few
    /// milliseconds. Compositor thread only.
    func frame(at time: TimeInterval) -> [HandPose] {
        lock.lock()
        let provider = running ? provider : nil
        lock.unlock()
        frames += 1
        var out = [HandPose(side: .left), HandPose(side: .right)]
        guard let provider else {
            noteStates(out)
            return out
        }
        for latest in provider.latestAnchors {
            let side = Self.side(of: latest)
            if lastSide[latest.id] != side {
                lastSide[latest.id] = side
                let held = latest.heldChirality.map { "\($0)" } ?? "unknown"
                Self.log(
                    "hand \(side == .left ? "left" : "right"): \"\(latest.accessory.name)\", held \(held), "
                        + "inherent \(latest.accessory.inherentChirality)"
                )
            }
            let published = Self.pose(of: latest, side: side)
            Self.publish(published)
            let shown = provider.predictAnchor(for: latest, at: time) ?? latest
            out[side.rawValue] = Self.pose(of: shown, side: side)
        }
        noteStates(out)
        return out
    }

    /// Which hand aims, from `hand=` in controls.txt (the runtime's pad).
    static var aimSide: HandPose.Side { rt_host_aim_hand() != 0 ? .left : .right }

    private func noteStates(_ poses: [HandPose]) {
        for pose in poses where lastState[pose.side.rawValue] != pose.state {
            lastState[pose.side.rawValue] = pose.state
            Self.log(
                "hand \(pose.side == .left ? "left" : "right") tracking: \(Self.stateName(pose.state)) "
                    + "(compositor frame \(frames))"
            )
        }
    }

    /// Held chirality when ARKit knows which hand holds it, else the one the
    /// accessory is made for; an unspecified one is taken as the right.
    private static func side(of anchor: AccessoryAnchor) -> HandPose.Side {
        let chirality = anchor.heldChirality ?? anchor.accessory.inherentChirality
        return chirality == .left ? .left : .right
    }

    private static func pose(of anchor: AccessoryAnchor, side: HandPose.Side) -> HandPose {
        var pose = HandPose()
        pose.side = side
        pose.source = .controller
        pose.state = stateCode(anchor.trackingState)
        pose.valid = pose.state != 0
        let locations = anchor.accessory.locations
        let origin = anchor.originFromAnchorTransform
        pose.originFromGrip =
            locations.contains(.grip)
            ? anchor.coordinateSpace(for: .grip, correction: .rendered).ancestorFromSpaceTransformFloat().matrix : origin
        pose.originFromAim =
            locations.contains(.aim)
            ? anchor.coordinateSpace(for: .aim, correction: .rendered).ancestorFromSpaceTransformFloat().matrix : origin
        pose.velocity = anchor.velocity
        pose.angularVelocity = anchor.angularVelocity
        pose.timestamp = anchor.timestamp
        return pose
    }

    private static func publish(_ pose: HandPose) {
        var grip = pose.originFromGrip
        var aim = pose.originFromAim
        var velocity = [pose.velocity.x, pose.velocity.y, pose.velocity.z]
        var angular = [pose.angularVelocity.x, pose.angularVelocity.y, pose.angularVelocity.z]
        let side = Int32(pose.side.rawValue)
        withUnsafeBytes(of: &grip) { g in
            withUnsafeBytes(of: &aim) { a in
                let gp = g.bindMemory(to: Float.self).baseAddress
                let ap = a.bindMemory(to: Float.self).baseAddress
                rt_host_hand_pose(side, pose.state, gp, ap, &velocity, &angular, pose.timestamp)
                metal_d3d9_immersive_hand_pose(side, pose.state, gp, ap, pose.timestamp)
            }
        }
    }

    private static func stateCode(_ state: AccessoryAnchor.TrackingState) -> Int32 {
        switch state {
        case .untracked: 0
        case .orientationTracked: 1
        case .positionOrientationTracked: 2
        case .positionOrientationTrackedLowAccuracy: 3
        @unknown default: 0
        }
    }

    private static func stateName(_ code: Int32) -> String {
        switch code {
        case 1: "orientation only"
        case 2: "position and orientation"
        case 3: "position and orientation, low accuracy"
        case -1: "not seen"
        default: "untracked"
        }
    }

    private static func label(_ device: GCDevice) -> String {
        let kind = device is GCSpatialAccessory ? "spatial accessory" : "controller"
        return "\(kind) \(device.vendorName ?? "unnamed") [\(device.productCategory)]"
    }

    private static func log(_ message: String) {
        rt_host_log("hands: " + message)
    }
}

private struct DeviceBox: @unchecked Sendable {
    let device: GCDevice
}
