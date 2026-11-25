import Foundation
import simd

// What a hand is doing with the controller in it, and the pose that follows
// from it. No app, game or rendering here: the app writes the controls as
// it reads its controllers and says what is in the hand; this turns that
// into a pose on any rig (`HandRig`) each frame.

/// What one hand's controller reports, each 0 to 1.
struct HandControlState: Sendable, Equatable {
    var trigger: Float = 0
    var grip: Float = 0
    /// The thumb on the stick or a face button, touched or pressed.
    var thumb: Float = 0
}

/// Both hands' controls, written by whoever polls the controllers and read
/// by whoever draws, on any threads.
final class HandControls: @unchecked Sendable {
    static let shared = HandControls()

    private let lock = NSLock()
    private var states = [HandControlState(), HandControlState()]
    private var seen = [false, false]

    /// side 0 left, 1 right.
    func set(side: Int, _ state: HandControlState) {
        guard side == 0 || side == 1 else { return }
        lock.lock()
        states[side] = state
        seen[side] = true
        lock.unlock()
    }

    /// The latest, and whether that side has ever been read.
    func state(side: Int) -> (HandControlState, Bool) {
        lock.lock()
        defer { lock.unlock() }
        return (states[side], seen[side])
    }
}

/// What a hand is holding, which picks the grip it closes into.
enum HandHold: Int, Sendable {
    case nothing, pistol, rifle, melee, support, point
}

/// One hand's pose over time: the controls eased with a short time
/// constant, so a pressed trigger closes the finger in a few frames rather
/// than snapping it, and the hold eased the same way when it changes.
struct HandPoseBlender: Sendable {
    /// Seconds to cover about two thirds of a change.
    var timeConstant: Float = 0.05
    private(set) var eased = HandControlState()
    private(set) var shape = HandShape.relaxed
    private var started = false
    private var lastHold = HandHold.nothing
    private var from = HandShape.relaxed
    private var change: Float = 1

    init(timeConstant: Float = 0.05) { self.timeConstant = timeConstant }

    /// The shape a hold rests in before the controls bend it.
    static func base(for hold: HandHold) -> HandShape {
        switch hold {
        case .nothing: return .relaxed
        case .pistol: return .pistol
        case .rifle: return .rifle
        case .melee: return .melee
        case .support: return .support
        case .point: return .point
        }
    }

    /// The shape for these controls and this hold, eased by dt seconds: the
    /// index follows the trigger towards a pull, the other three the grip
    /// towards a fist, the thumb its touch towards the top of the
    /// controller. Pointing keeps the index straight unless the trigger is
    /// all the way in (a click). Holding a weapon, the three already wrap
    /// it, so the grip button tightens them only a little further.
    mutating func step(controls: HandControlState, hold: HandHold, dt: Float) -> HandShape {
        let first = !started
        let k = first ? 1 : 1 - exp(-max(dt, 0) / max(timeConstant, 1e-3))
        started = true
        if first { lastHold = hold }
        eased.trigger += (controls.trigger - eased.trigger) * k
        eased.grip += (controls.grip - eased.grip) * k
        eased.thumb += (controls.thumb - eased.thumb) * k
        var target = Self.base(for: hold)
        switch hold {
        case .point:
            target = target.with(finger: 1, from: .trigger, by: max(0, eased.trigger - 0.6) / 0.4 * 0.5)
        default:
            target = target.with(finger: 1, from: .trigger, by: eased.trigger)
        }
        let grip = hold == .nothing || hold == .point ? eased.grip : eased.grip * 0.3
        for finger in 2...4 { target = target.with(finger: finger, from: .fist, by: grip) }
        target = target.with(finger: 0, from: .thumbDown, by: eased.thumb)
        // A change of hold eased over a few time constants from where the
        // hand was; otherwise the controls' easing is all there is.
        if hold != lastHold {
            lastHold = hold
            from = shape
            change = 0
        }
        change = min(1, change + k * 0.5)
        shape = change < 1 ? HandShape.mix(from, target, change) : target
        return shape
    }

    /// The pose a model's file names for a hold (`HandPoseFile`): its own
    /// grips, pointing in menus, relaxed otherwise.
    static func poseName(for hold: HandHold) -> String {
        switch hold {
        case .nothing: return "relaxed"
        case .pistol, .rifle: return "pistol_grip"
        case .support: return "rifle_grip"
        case .melee: return "melee_grip"
        case .point: return "point"
        }
    }

    private var lastPose: HandJointPose?
    private var fromPose: HandJointPose?

    /// The same from a model's own poses, blended per joint: the hold's
    /// pose, the index towards "trigger" by the trigger, the other three
    /// towards "grip" by the grip, the thumb half way towards "fist" by
    /// its touch. nil when the file lacks the hold's pose; the curl shapes
    /// (`step(controls:hold:dt:)`) are the fallback.
    mutating func step(
        controls: HandControlState, hold: HandHold, dt: Float, file: HandPoseFile, rig: HandRig,
        gripCurl: Float = 0.85, thumbWrap: Float = 0.7
    ) -> HandJointPose? {
        guard let base = file.pose(named: Self.poseName(for: hold), on: rig) else { return nil }
        let first = !started
        let k = first ? 1 : 1 - exp(-max(dt, 0) / max(timeConstant, 1e-3))
        started = true
        if first { lastHold = hold }
        eased.trigger += (controls.trigger - eased.trigger) * k
        eased.grip += (controls.grip - eased.grip) * k
        eased.thumb += (controls.thumb - eased.thumb) * k
        let fingers = rig.names.map { HandJoint(name: $0)?.finger }
        var target = base
        let pull = hold == .point ? max(0, eased.trigger - 0.6) / 0.4 * 0.5 : eased.trigger
        // On a gun the three lower fingers stay closed round its grip and
        // the thumb over it, as far as the glove's grip says, whatever the
        // grip button; the index rests on the trigger and follows it.
        let onGun = hold == .pistol || hold == .rifle
        let squeeze = hold == .nothing || hold == .point ? eased.grip : eased.grip * 0.3
        if let trigger = file.pose(named: "trigger", on: rig) {
            target = target.blended(to: trigger, weights: fingers.map { $0 == 1 ? pull : 0 })
        }
        if let fist = file.pose(named: "fist", on: rig), onGun {
            target = target.blended(to: fist, weights: fingers.map { f in
                guard let f else { return 0 }
                return f >= 2 ? gripCurl : f == 0 ? max(thumbWrap, eased.thumb * 0.5) : 0
            })
        } else {
            if let grip = file.pose(named: "grip", on: rig) {
                target = target.blended(to: grip, weights: fingers.map { ($0 ?? 0) >= 2 ? squeeze : 0 })
            }
            if let fist = file.pose(named: "fist", on: rig) {
                target = target.blended(to: fist, weights: fingers.map { $0 == 0 ? eased.thumb * 0.5 : 0 })
            }
        }
        if hold != lastHold {
            lastHold = hold
            fromPose = lastPose
            change = 0
        }
        change = min(1, change + k * 0.5)
        let out = change < 1 && fromPose != nil ? fromPose!.blended(to: target, by: change) : target
        lastPose = out
        return out
    }
}
