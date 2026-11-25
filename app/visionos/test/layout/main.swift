// The layout edit's math (Host/Layout/HUDLayout.swift), on the Mac:
//   swiftc -O app/visionos/Host/Layout/HUDLayout.swift app/visionos/test/layout/main.swift -o layout_test
// A piece picked up and put down where it was keeps its values, turns
// read back to the angles they were built from, mirrored sides agree,
// and a piece carried by a controller moves with it.
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

func near(_ a: SIMD3<Float>, _ b: SIMD3<Float>, _ tolerance: Float = 1e-3) -> Bool { simd_length(a - b) < tolerance }

func near(_ a: simd_float3x3, _ b: simd_float3x3) -> Bool {
    near(a.columns.0, b.columns.0) && near(a.columns.1, b.columns.1) && near(a.columns.2, b.columns.2)
}

let angles: [SIMD3<Float>] = [[0, 0, 0], [40, 0, 0], [25, -30, 10], [-60, 75, -45], [10, 170, 5], [80, 20, -120]]

for mirrored in [false, true] {
    for a in angles {
        let m = HUDLayoutMath.orbTurn(a, mirrored: mirrored)
        let back = HUDLayoutMath.orbDegrees(m, mirrored: mirrored)
        check(near(HUDLayoutMath.orbTurn(back, mirrored: mirrored), m), "an orb turned \(a)\(mirrored ? " mirrored" : "") reads back")
        let g = HUDLayoutMath.gloveTurn(a, mirrored: mirrored)
        let gBack = HUDLayoutMath.gloveDegrees(g, mirrored: mirrored)
        check(near(HUDLayoutMath.gloveTurn(gBack, mirrored: mirrored), g), "a glove turned \(a)\(mirrored ? " mirrored" : "") reads back")
    }
}

// A glove's fitting in and out, both hands.
for mirrored in [false, true] {
    let glove = HUDLayout.Glove(offset: [0.03, -0.005, 0.06], turn: [-15, 10, 5], scale: 1.15)
    let fit = HUDLayoutMath.gloveFit(glove, mirrored: mirrored)
    let back = HUDLayoutMath.glove(fromFit: fit, scale: glove.scale, mirrored: mirrored)
    check(near(back.offset, glove.offset) && near(back.turn, glove.turn, 0.05), "a glove's fitting reads back\(mirrored ? " (left)" : "")")
}

// Mirrored left and right gloves with the same values are reflections
// of each other across the grip's x.
let glove = HUDLayout.Glove(offset: [0.03, 0.01, 0.06], turn: [-15, 20, 10], scale: 1)
let right = HUDLayoutMath.gloveFit(glove, mirrored: false), left = HUDLayoutMath.gloveFit(glove, mirrored: true)
let reflect = simd_float4x4(diagonal: [-1, 1, 1, 1])
check(near(SIMD3(left.columns.3.x, left.columns.3.y, left.columns.3.z), [-0.03, 0.01, 0.06]), "the left glove sits mirrored")
let reflected = reflect * right * reflect
check(near(simd_float3x3(SIMD3(reflected.columns.0.x, reflected.columns.0.y, reflected.columns.0.z),
                         SIMD3(reflected.columns.1.x, reflected.columns.1.y, reflected.columns.1.z),
                         SIMD3(reflected.columns.2.x, reflected.columns.2.y, reflected.columns.2.z)),
           simd_float3x3(SIMD3(left.columns.0.x, left.columns.0.y, left.columns.0.z),
                         SIMD3(left.columns.1.x, left.columns.1.y, left.columns.1.z),
                         SIMD3(left.columns.2.x, left.columns.2.y, left.columns.2.z))),
      "the left glove's turn is the right's reflected")

// An orb at a hand: placed as the compositor does, read back.
let grip = SIMD3<Float>(0.2, 1.1, -0.4)
let axes = simd_float3x3(SIMD3(1, 0, 0), SIMD3(0, 1, 0), SIMD3(0, 0, 1))
let anchor = HUDLayoutMath.Frame(origin: grip, axes: axes)
for (hand, out) in [(0, Float(-1)), (1, Float(1))] {
    let orb = HUDLayout.Orb(offset: [0.09, 0.07, -0.02], turn: [40, 10, -5], size: 0.08)
    let centre = grip + SIMD3(out * orb.offset.x, orb.offset.y, orb.offset.z)
    var facing = SIMD3<Float>(0, 1.6, 0) - centre
    facing.y = 0
    facing = simd_normalize(facing)
    let base = HUDLayoutMath.facingFrame(facing: facing)
    let orient = base * HUDLayoutMath.orbTurn(orb.turn, mirrored: hand == 0)
    let world = simd_float4x4(SIMD4(orient.columns.0, 0), SIMD4(orient.columns.1, 0), SIMD4(orient.columns.2, 0), SIMD4(centre, 1))
    let back = HUDLayoutMath.orb(at: world, anchor: anchor, facing: base, size: orb.size, mirrored: hand == 0)
    check(near(back.offset, orb.offset) && near(back.turn, orb.turn, 0.05), "an orb at the \(hand == 0 ? "left" : "right") hand reads back")
}

// Picked up and carried: the piece keeps its place against the
// controller, so moving the controller moves it by the same.
let controller = simd_float4x4(simd_quatf(angle: 0.4, axis: simd_normalize(SIMD3(0.2, 1, 0.1))))
    * simd_float4x4(diagonal: [1, 1, 1, 1])
var at = controller
at.columns.3 = [0.1, 1.2, -0.3, 1]
var piece = matrix_identity_float4x4
piece.columns.3 = [0.25, 1.15, -0.35, 1]
let hold = HUDLayoutMath.hold(piece: piece, controller: at)
check(near(HUDLayoutMath.Frame(HUDLayoutMath.held(hold, controller: at)).origin, [0.25, 1.15, -0.35]), "a piece picked up stays put until moved")
var moved = at
moved.columns.3 += [0.05, -0.02, 0.1, 0]
check(near(HUDLayoutMath.Frame(HUDLayoutMath.held(hold, controller: moved)).origin, [0.30, 1.13, -0.25]), "a carried piece moves with the controller")

// Values are kept to their ranges.
var wild = HUDLayout(orbs: [HUDLayout.Orb(offset: [9, 0, 0], turn: .zero, size: 4), HUDLayout.Orb(offset: .zero, turn: .zero, size: 0)],
                     bodyOrbs: [.zero, .zero], gloves: [HUDLayout.Glove(offset: .zero, turn: .zero, scale: 9),
                                                         HUDLayout.Glove(offset: .zero, turn: .zero, scale: 0)],
                     reticleDistance: 100, reticleScale: 0)
wild.clamp()
check(wild.orbs[0].size == HUDLayout.orbSizeRange.upperBound && wild.orbs[1].size == HUDLayout.orbSizeRange.lowerBound
      && wild.orbs[0].offset.x == HUDLayout.offsetLimit && wild.gloves[0].scale == 2 && wild.gloves[1].scale == 0.5
      && wild.reticleDistance == 30 && wild.reticleScale == 0.25, "values are held to their ranges")


// A held orb turns with the controller, and the stick across tilts it.
do {
    let base = HUDLayoutMath.facingFrame(facing: [0, 0, 1])
    let anchor = HUDLayoutMath.Frame(origin: [0.2, 1.0, -0.3], axes: matrix_identity_float3x3)
    let start = HUDLayout.Orb(offset: [0.09, 0.07, 0], turn: [40, 0, 0], size: 0.08)
    let r = base * HUDLayoutMath.orbTurn(start.turn, mirrored: false)
    let world = simd_float4x4(SIMD4(r.columns.0, 0), SIMD4(r.columns.1, 0), SIMD4(r.columns.2, 0), SIMD4(0.29, 1.07, -0.3, 1))
    let controller = simd_float4x4(SIMD4(1, 0, 0, 0), SIMD4(0, 1, 0, 0), SIMD4(0, 0, 1, 0), SIMD4(-0.2, 1.0, -0.3, 1))
    var hold = HUDLayoutMath.hold(piece: world, controller: controller)
    let twist = simd_float4x4(simd_quatf(angle: 20 * .pi / 180, axis: [0, 0, 1])) // a roll of the wrist
    var turned = controller * twist
    turned.columns.3 = controller.columns.3
    let rolled = HUDLayoutMath.orb(at: HUDLayoutMath.held(hold, controller: turned), anchor: anchor, facing: base,
                                   size: start.size, mirrored: false)
    check(abs(rolled.turn.x - start.turn.x) > 1 || abs(rolled.turn.y) > 1 || abs(rolled.turn.z) > 1,
          "turning the controller turns a held orb (\(rolled.turn))")
    for _ in 0..<60 { hold = HUDLayoutMath.tilted(hold, piece: .orb(hand: 1), stickX: 1, dt: 1.0 / 60) }
    let tilted = HUDLayoutMath.orb(at: HUDLayoutMath.held(hold, controller: controller), anchor: anchor, facing: base,
                                   size: start.size, mirrored: false)
    check(abs(tilted.turn.x - (start.turn.x + HUDLayout.tiltRate)) < 0.5 && abs(tilted.turn.y) < 0.5 && abs(tilted.turn.z) < 0.5,
          "a second of stick right leans an orb's top back by \(HUDLayout.tiltRate) degrees (\(tilted.turn))")
    check(near(HUDLayoutMath.Frame(HUDLayoutMath.held(hold, controller: controller)).origin, [0.29, 1.07, -0.3]),
          "tilting leaves the orb where it is")
    var gHold = matrix_identity_float4x4
    for _ in 0..<60 { gHold = HUDLayoutMath.tilted(gHold, piece: .glove(hand: 1), stickX: -1, dt: 1.0 / 60) }
    let g = HUDLayoutMath.glove(fromFit: gHold, scale: 1, mirrored: false)
    check(abs(g.turn.x + HUDLayout.tiltRate) < 0.5, "the stick tilts a glove's pitch too (\(g.turn))")
    var old = wild
    old.bodyTurns = nil
    check(old.bodyTurn(1) == old.orbs[1].turn, "a layout saved before third person turns shares the hand's")
}


// The gizmo: handles are found by hand and by ray, and each drags only
// its own part.
do {
    let g = Gizmo(centre: [0, 1, -0.5], axes: matrix_identity_float3x3, radius: 0.04, mode: .move)
    let tip = g.centre + SIMD3(1, 0, 0) * (g.arrowStart + g.arrowLength * 0.8)
    check(g.hit(point: tip, rayOrigin: nil, rayDirection: nil) == .axis(0), "a hand at the red arrow takes it")
    check(g.hit(point: g.centre + SIMD3(0, 0.5, 0), rayOrigin: nil, rayDirection: nil) == nil, "a hand far off takes nothing")
    let up = g.centre + SIMD3(0, 1, 0) * (g.arrowStart + g.arrowLength / 2)
    check(g.hit(point: nil, rayOrigin: [0, up.y, 0.5], rayDirection: [0, 0, -1]) == .axis(1), "a ray at the green arrow takes it")
    let r = Gizmo(centre: g.centre, axes: g.axes, radius: g.radius, mode: .rotate)
    check(r.hit(point: r.centre + SIMD3(0, r.ringRadius, 0), rayOrigin: nil, rayDirection: nil) == .ring(0)
          || r.hit(point: r.centre + SIMD3(0, r.ringRadius, 0), rayOrigin: nil, rayDirection: nil) == .ring(2),
          "a hand on a ring takes a ring")
    let sc = Gizmo(centre: g.centre, axes: g.axes, radius: g.radius, mode: .scale)
    check(sc.hit(point: sc.scalePoint, rayOrigin: nil, rayDirection: nil) == .scale, "a hand on the square takes it")
    check(abs(Gizmo.slide(from: .zero, to: [0.123, 0.5, 0.2], axis: [1, 0, 0], snap: false) - 0.123) < 1e-5,
          "an arrow drag counts only motion along its axis")
    check(abs(Gizmo.slide(from: .zero, to: [0.123, 0, 0], axis: [1, 0, 0], snap: true) - 0.12) < 1e-5, "snapped to a centimetre")
    let q0 = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
    let arc = Gizmo.turn(centre: .zero, axis: [0, 1, 0], startPoint: [0.1, 0, 0], point: [0, 0, -0.1],
                         startTurn: q0, nowTurn: q0, snap: false)
    check(abs(arc - .pi / 2) < 1e-3, "a quarter arc round the up axis turns a quarter (\(arc))")
    let twist = Gizmo.turn(centre: .zero, axis: [0, 0, 1], startPoint: .zero, point: .zero,
                           startTurn: q0, nowTurn: simd_quatf(angle: 0.3, axis: [0, 0, 1]), snap: false)
    check(abs(twist - 0.3) < 1e-3, "a wrist twist on the axis turns by the twist")
    let snapped = Gizmo.turn(centre: .zero, axis: [0, 0, 1], startPoint: .zero, point: .zero,
                             startTurn: q0, nowTurn: simd_quatf(angle: 0.3, axis: [0, 0, 1]), snap: true)
    check(abs(snapped - 15 * .pi / 180) < 1e-4, "a twist snaps to 5 degrees")
    let f = Gizmo.scaleFactor(from: .zero, to: [0.02, 0.02, 0], outward: [1, 1, 0], radius: 0.04, snap: false)
    check(f > 1.6 && f < 1.8, "dragging the square outwards grows (\(f))")
    check(GizmoMode.move.next == .rotate && GizmoMode.scale.next == .move, "the modes cycle")
}

// The glove in the gun's frame moves with the gun as one piece: the gun
// as the game draws it is built here the runtime's way, from the tilted
// aim's yaw and pitch (headlook.c rt_headlook_yaw_pitch) and its roll
// (rt_headlook_roll), placed at the aim less the rest; for random hand
// poses with any roll, the glove seen from that gun never changes.
do {
    func gameGun(aim: simd_float4x4, tilt: Float, headYaw: Float, rest: SIMD3<Float>, upm: Float) -> simd_float4x4 {
        let t = -tilt * .pi / 180
        var turn = matrix_identity_float4x4
        turn.columns.1 = [0, cos(t), sin(t), 0]
        turn.columns.2 = [0, -sin(t), cos(t), 0]
        let m = aim * turn
        let f = -SIMD3<Double>(Double(m.columns.2.x), Double(m.columns.2.y), Double(m.columns.2.z))
        let yaw = atan2(-f.x, -f.z), pitch = asin(min(max(f.y, -1), 1))
        var r0 = SIMD3<Double>(-f.z, 0, f.x)
        r0 /= simd_length(r0)
        let u0 = simd_cross(r0, f)
        let u = SIMD3<Double>(Double(m.columns.1.x), Double(m.columns.1.y), Double(m.columns.1.z))
        let roll = atan2(simd_dot(u, r0), simd_dot(u, u0))
        // Yaw left about up, pitch up about the level right, roll right
        // side down about forward.
        let fw = SIMD3<Double>(-sin(yaw) * cos(pitch), sin(pitch), -cos(yaw) * cos(pitch))
        let rl = SIMD3<Double>(cos(yaw), 0, -sin(yaw))
        let ul = simd_cross(rl, fw)
        let up = ul * cos(roll) + rl * sin(roll), right = rl * cos(roll) - ul * sin(roll)
        let hf = SIMD3<Float>(-sin(headYaw), 0, -cos(headYaw)), hr = SIMD3<Float>(cos(headYaw), 0, -sin(headYaw))
        let p = SIMD3(aim.columns.3.x, aim.columns.3.y, aim.columns.3.z) - (rest.x * hf + rest.y * hr + SIMD3(0, rest.z, 0)) / upm
        return simd_float4x4(SIMD4(SIMD3<Float>(right), 0), SIMD4(SIMD3<Float>(up), 0), SIMD4(SIMD3<Float>(-fw), 0), SIMD4(p, 1))
    }
    var rng = SystemRandomNumberGenerator()
    let fit = HUDLayoutMath.gloveFit(HUDLayout.Glove(offset: [0.03, -0.02, 0.06], turn: [-20, 10, 5], scale: 1.15), mirrored: false)
    var first: simd_float4x4?
    var worst: Float = 0
    for _ in 0..<300 {
        let q = simd_quatf(angle: Float.random(in: 0..<(2 * .pi), using: &rng),
                           axis: simd_normalize(SIMD3(Float.random(in: -1...1, using: &rng), Float.random(in: -1...1, using: &rng),
                                                      Float.random(in: -1...1, using: &rng)) + [0, 0.001, 0]))
        var aim = simd_float4x4(q)
        aim.columns.3 = SIMD4(Float.random(in: -1...1, using: &rng), Float.random(in: 0.5...1.8, using: &rng),
                              Float.random(in: -1...1, using: &rng), 1)
        let hy = Float.random(in: -.pi ... .pi, using: &rng)
        // Not within a degree of straight up or down, where yaw is lost.
        let fz = -(aim * simd_float4x4(simd_quatf(angle: -25 * .pi / 180, axis: [1, 0, 0]))).columns.2.y
        if abs(fz) > 0.9998 { continue }
        let glove = HUDLayoutMath.gunFrame(aim: aim, tiltDegrees: 25, headYaw: hy, rest: [0, 0, -3], unitsPerMetre: 39.37) * fit
        let rel = gameGun(aim: aim, tilt: 25, headYaw: hy, rest: [0, 0, -3], upm: 39.37).inverse * glove
        if let f = first {
            for c in 0..<4 { worst = max(worst, simd_length(rel[c] - f[c])) }
        } else { first = rel }
    }
    check(worst < 1e-3, "a glove in the gun's frame keeps its place on the gun at 300 hand poses with any roll (worst \(worst))")
    // A glove placed from the grip and turned into the gun's frame stays
    // where it was in the room.
    var grip = matrix_identity_float4x4
    grip.columns.3 = [0.1, 1.2, -0.3, 1]
    let aim = grip * simd_float4x4(simd_quatf(angle: 0.4, axis: [1, 0, 0]))
    let gun = HUDLayoutMath.gunFrame(aim: aim, tiltDegrees: 25, headYaw: 0.3, rest: [0, 0, -3], unitsPerMetre: 39.37)
    let g = HUDLayout.Glove(offset: [0.04, 0.07, 0.08], turn: [-34, 8, 1], scale: 1.15)
    let moved = HUDLayoutMath.reanchor(g, mirrored: false, from: grip, to: gun)
    let before = grip * HUDLayoutMath.gloveFit(g, mirrored: false), after = gun * HUDLayoutMath.gloveFit(moved, mirrored: false)
    var d: Float = 0
    for c in 0..<4 { d = max(d, simd_length(before[c] - after[c])) }
    check(d < 1e-3, "a glove placed from the grip keeps its place in the room when measured from the gun (\(d))")
}

// Round trip through JSON.
import Foundation
let data = try! JSONEncoder().encode(wild)
check((try? JSONDecoder().decode(HUDLayout.self, from: data)) == wild, "a layout saves and loads unchanged")

print(failed == 0 ? "ALL PASS: \(passed) passed, 0 failed" : "FAILED: \(passed) passed, \(failed) failed")
exit(failed == 0 ? 0 : 1)
