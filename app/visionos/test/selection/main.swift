// The soft target lock (Host/Selection/SoftTarget.swift), on the Mac:
//   swiftc -O app/visionos/Host/Selection/*.swift app/visionos/test/selection/main.swift -o selection_test
// Cone entry and exit, hysteresis, ties, cycling, a target dying and an
// empty set, with the aim looking down -Z from the origin and Y up.
import Foundation
import simd

var failed = 0
var passed = 0

func check(_ ok: Bool, _ what: String) {
    if ok {
        passed += 1
        print("ok   \(what)")
    } else {
        failed += 1
        print("FAIL \(what)")
    }
}

let aim = SoftTargetRay(origin: .zero, direction: [0, 0, -1])

/// A candidate at `degrees` to the right of the aim (negative is left),
/// `distance` away, on the horizon.
func at(_ id: Int, _ degrees: Float, _ distance: Float = 10, radius: Float = 0, selectable: Bool = true) -> SoftTargetCandidate {
    let r = degrees * .pi / 180
    return SoftTargetCandidate(id: id, centre: [sin(r) * distance, 0, -cos(r) * distance],
                               radius: radius, selectable: selectable)
}

/// The aim turned `degrees` to the right.
func aimed(_ degrees: Float) -> SoftTargetRay {
    let r = degrees * .pi / 180
    return SoftTargetRay(origin: .zero, direction: [sin(r), 0, -cos(r)])
}

// Empty and nothing in range.
do {
    var s = SoftTargetSelector()
    check(s.update(time: 0, aim: aim, candidates: []) == nil, "an empty set selects nothing")
    check(s.strength == 0, "an empty set has no highlight")
    check(s.update(time: 0.1, aim: aim, candidates: [at(1, 40), at(2, -25)]) == nil, "candidates outside the cone select nothing")
    check(s.update(time: 0.2, aim: aim, candidates: [at(3, 0, selectable: false)]) == nil, "an unselectable candidate is never picked")
    let behind = SoftTargetCandidate(id: 4, centre: [0, 0, 5], radius: 0, selectable: true)
    check(s.update(time: 0.3, aim: aim, candidates: [behind]) == nil, "a candidate behind the aim is ignored")
    var near = SoftTargetSelector(config: SoftTargetConfig(maxDistance: 20))
    check(near.update(time: 0, aim: aim, candidates: [at(5, 0, 30)]) == nil, "a candidate beyond the range is ignored")
}

// Cone entry and exit.
do {
    var s = SoftTargetSelector()
    check(s.update(time: 0, aim: aim, candidates: [at(1, 12)]) == nil, "12 degrees is outside a 10 degree cone")
    check(s.update(time: 0.1, aim: aim, candidates: [at(1, 8)]) == 1, "8 degrees enters the cone")
    check(s.update(time: 0.2, aim: aim, candidates: [at(1, 12)]) == 1, "the held one stays at 12, inside the exit cone")
    check(s.update(time: 0.3, aim: aim, candidates: [at(1, 14)]) == nil, "it is dropped past the 13 degree exit cone")
    check(s.update(time: 0.4, aim: aim, candidates: [at(1, 12)]) == nil, "and does not come back until it re-enters at 10")
    var wide = SoftTargetSelector()
    check(wide.update(time: 0, aim: aim, candidates: [at(2, 12, 10, radius: 1)]) == 2,
          "a big candidate counts by its edge, not its centre")
}

// The strength rises after a pick and falls after a loss.
do {
    var s = SoftTargetSelector()
    s.update(time: 0, aim: aim, candidates: [at(1, 0)])
    check(s.strength == 0, "the highlight starts at zero on a new pick")
    s.update(time: 0.06, aim: aim, candidates: [at(1, 0)])
    check(abs(s.strength - 0.5) < 0.01, "half way up after half the rise time")
    s.update(time: 0.5, aim: aim, candidates: [at(1, 0)])
    check(s.strength == 1, "fully up after the rise time")
    s.update(time: 0.6, aim: aim, candidates: [])
    check(s.selected == nil && s.fading == 1 && s.strength > 0 && s.strength < 1, "fades on the lost one")
    s.update(time: 1.0, aim: aim, candidates: [])
    check(s.strength == 0, "and is gone after the fall time")
}

// Hysteresis.
do {
    var s = SoftTargetSelector()
    s.update(time: 0, aim: aim, candidates: [at(1, 1), at(2, 6)])
    check(s.selected == 1, "the nearer one to the aim is picked")
    // The aim drifts so 2 is just nearer: within the margin, nothing changes.
    let a = aimed(4)
    s.update(time: 0.1, aim: a, candidates: [at(1, 1), at(2, 6)])
    s.update(time: 0.5, aim: a, candidates: [at(1, 1), at(2, 6)])
    check(s.selected == 1, "a challenger within the 2 degree margin never takes over")
    // Further, past the margin: it takes over only after the hold time.
    let b = aimed(6)
    s.update(time: 1.0, aim: b, candidates: [at(1, 1), at(2, 6)])
    check(s.selected == 1, "a challenger past the margin does not take over at once")
    s.update(time: 1.1, aim: b, candidates: [at(1, 1), at(2, 6)])
    check(s.selected == 1, "nor after 100 ms")
    s.update(time: 1.16, aim: b, candidates: [at(1, 1), at(2, 6)])
    check(s.selected == 2, "it takes over after 150 ms")
    // A challenger that wavers starts its clock again.
    s.update(time: 2.0, aim: aimed(0), candidates: [at(1, 1), at(2, 6)])
    s.update(time: 2.1, aim: aimed(0), candidates: [at(1, 1), at(2, 6)])
    s.update(time: 2.2, aim: aimed(0), candidates: [at(1, 1), at(2, 6)])
    check(s.selected == 1, "back on 1 after a full hold")
    s.update(time: 2.25, aim: aimed(6), candidates: [at(1, 1), at(2, 6)])
    s.update(time: 2.3, aim: aimed(3.5), candidates: [at(1, 1), at(2, 6)])
    s.update(time: 2.35, aim: aimed(6), candidates: [at(1, 1), at(2, 6)])
    s.update(time: 2.45, aim: aimed(6), candidates: [at(1, 1), at(2, 6)])
    check(s.selected == 1, "an interrupted challenge restarts its clock")
}

// Ties.
do {
    var s = SoftTargetSelector()
    check(s.update(time: 0, aim: aim, candidates: [at(1, 3, 20), at(2, -3, 8)]) == 2,
          "at the same angle the nearer one wins")
    var t = SoftTargetSelector()
    check(t.update(time: 0, aim: aim, candidates: [at(7, 3, 10), at(4, -3, 10)]) == 4,
          "at the same angle and distance the lower id wins")
    var u = SoftTargetSelector()
    check(u.update(time: 0, aim: aim, candidates: [at(4, -3, 10), at(7, 3, 10)]) == 4,
          "whatever order the candidates come in")
}

// A target dying, and a stable id while it moves.
do {
    var s = SoftTargetSelector()
    s.update(time: 0, aim: aim, candidates: [at(1, 0), at(2, 7)])
    check(s.selected == 1, "picked before it moves")
    s.update(time: 0.1, aim: aim, candidates: [at(2, 7), at(1, 5, 15)])
    check(s.selected == 1, "the same id is held while it moves")
    s.update(time: 0.2, aim: aim, candidates: [at(1, 5, 15, selectable: false), at(2, 7)])
    check(s.selected == 2, "when it dies the next one takes over at once")
    s.update(time: 0.3, aim: aim, candidates: [at(2, 7)])
    check(s.selected == 2, "and is held")
    s.update(time: 0.4, aim: aim, candidates: [])
    check(s.selected == nil, "when everything is gone nothing is selected")
}

// Cycling, in screen order.
do {
    var s = SoftTargetSelector()
    let set = [at(1, -20), at(2, -4), at(3, 2), at(4, 15), at(5, 50)]
    s.update(time: 0, aim: aim, candidates: set)
    check(s.selected == 3, "the start is the nearest to the aim")
    check(s.cycle(.right) == 4, "a flick right goes to the next on the right")
    check(s.cycle(.right) == 4, "and stays at the right end, ignoring one outside the cycle cone")
    check(s.cycle(.left) == 3, "a flick left steps back")
    check(s.cycle(.left) == 2, "and on")
    check(s.cycle(.left) == 1, "to the left end")
    check(s.cycle(.left) == 1, "and stays there")
    // A cycled pick is not challenged by one nearer the aim.
    s.update(time: 0.2, aim: aim, candidates: set)
    s.update(time: 1.0, aim: aim, candidates: set)
    check(s.selected == 1, "a cycled pick holds against a nearer one")
    // It is released when it leaves the cycle cone.
    s.update(time: 1.1, aim: aimed(20), candidates: set)
    check(s.selected != 1, "and is let go once it leaves the cycle cone")
    var e = SoftTargetSelector()
    e.update(time: 0, aim: aim, candidates: [at(1, -20), at(2, 15)])
    check(e.selected == nil && e.cycle(.right) == 2, "with nothing held a flick right takes the nearest on the right")
    var f = SoftTargetSelector()
    f.update(time: 0, aim: aim, candidates: [])
    check(f.cycle(.left) == nil, "a flick with nothing to pick does nothing")
}

// The gaze helps but is optional.
do {
    var s = SoftTargetSelector()
    let gaze = aimed(20)
    check(s.update(time: 0, aim: aim, gaze: gaze, candidates: [at(1, 21)]) == 1, "a candidate near the gaze is picked")
    var t = SoftTargetSelector()
    check(t.update(time: 0, aim: aim, candidates: [at(1, 21)]) == nil, "and not without the gaze")
}

print("selection: \(passed) passed, \(failed) failed")
exit(failed == 0 ? 0 : 1)
