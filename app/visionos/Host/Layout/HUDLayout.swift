import simd

/// Where each placed piece of the full view sits: the orbs at the hands
/// and beside the body, the gloves in their grips, the reticle out along
/// the aim. Plain values, saved as JSON by `LayoutStore` and changed live
/// by `LayoutEditor`, so this file is the whole of what an edit can
/// touch and builds alone for the offline test.
///
/// Values for a pair are kept per hand (index 0 left, 1 right) in the
/// right hand's sense: x is "outwards" for the orbs and the right hand's
/// x for the gloves, and the yaw and roll turn the same way as seen from
/// outside. The left hand applies them mirrored, so equal values give a
/// symmetric pair.
struct HUDLayout: Codable, Equatable {

    /// An orb at its hand: centre offset from the grip in the head's
    /// axes, metres (out, up, forward), its turn from upright facing the
    /// head in degrees (pitch up, yaw out, roll clockwise) and its
    /// diameter in metres.
    struct Orb: Codable, Equatable {
        var offset: SIMD3<Float>
        var turn: SIMD3<Float>
        var size: Float
    }

    /// A glove in its controller's grip: offset in centimetres' worth of
    /// metres (right, up, back), turn in degrees (pitch, yaw, roll) and
    /// model scale.
    struct Glove: Codable, Equatable {
        var offset: SIMD3<Float>
        var turn: SIMD3<Float>
        var scale: Float
        /// How far the three lower fingers close round a gun's grip, and
        /// the thumb over it, 0 open to 1 closed; absent until edited.
        var curl: Float?
        var thumb: Float?

        static let defaultCurl: Float = 0.85
        static let defaultThumb: Float = 0.7
        var gripCurl: Float { curl ?? Self.defaultCurl }
        var thumbWrap: Float { thumb ?? Self.defaultThumb }

        /// The grip's curl and thumb carried over from another glove.
        func keepingGrip(of g: Glove) -> Glove {
            var out = self
            out.curl = g.curl
            out.thumb = g.thumb
            return out
        }
    }

    var orbs: [Orb]
    /// The orbs in third person, from the head along the body's facing,
    /// metres (out, up, forward).
    var bodyOrbs: [SIMD3<Float>]
    /// Their turns in third person, as `Orb.turn`; absent in a layout
    /// saved before they had their own, which then shares the hand's.
    var bodyTurns: [SIMD3<Float>]?
    var gloves: [Glove]
    /// A glove's place per weapon, by the folder its first person model
    /// sits in (rifle_steyraug): with the hand aim the drawn gun follows
    /// the controller, so its grip sits at a fixed place from it, one per
    /// weapon. Absent in a layout saved before, and a weapon with none
    /// uses `gloves`.
    var weaponGloves: [String: [Glove]]?
    /// What the glove places are measured from: nil (a layout saved before)
    /// the controller's grip, "gun" the frame the game's gun is drawn in
    /// (`HUDLayoutMath.gunFrame`), so a glove set on a grip stays on it
    /// when the wrist twists. A grip layout is turned into gun places once,
    /// on the first frame both hands are tracked, and `builtGloves` keeps
    /// the built in places turned the same way for a reset.
    var gloveAnchor: String?
    var builtGloves: [Glove]?
    /// The first person reticle's distance out along the aim, metres.
    var reticleDistance: Float
    /// The reticle's size against its default, 1 as built.
    var reticleScale: Float

    /// What each piece's values may be, so a long stick hold or a wild
    /// grab cannot lose a piece or blow it up past use.
    static let orbSizeRange: ClosedRange<Float> = 0.03...0.30
    static let gloveScaleRange: ClosedRange<Float> = 0.5...2
    static let reticleScaleRange: ClosedRange<Float> = 0.25...4
    static let reticleDistanceRange: ClosedRange<Float> = 0.5...30
    static let offsetLimit: Float = 1.5
    /// A full stick held for a second changes a size by this fraction.
    static let stickRate: Float = 0.8
    /// A full stick across held for a second tilts a piece this far, degrees.
    static let tiltRate: Float = 45

    /// The glove on `hand` for the weapon in hand ("" for none).
    func glove(_ hand: Int, weapon: String) -> Glove {
        if !weapon.isEmpty, let g = weaponGloves?[weapon], g.count == 2 { return g[hand] }
        return gloves[hand]
    }

    /// Sets the glove on `hand` for the weapon in hand, or the general one
    /// with no weapon; a weapon's first change starts from what it showed.
    mutating func setGlove(_ hand: Int, weapon: String, _ g: Glove) {
        guard !weapon.isEmpty else { gloves[hand] = g; return }
        var pair = [glove(0, weapon: weapon), glove(1, weapon: weapon)]
        pair[hand] = g
        var all = weaponGloves ?? [:]
        all[weapon] = pair
        weaponGloves = all
    }

    /// Puts the glove on `hand` back to the general one for this weapon,
    /// dropping the weapon's own once neither hand differs.
    mutating func resetGlove(_ hand: Int, weapon: String, to general: Glove) {
        guard !weapon.isEmpty, var pair = weaponGloves?[weapon], pair.count == 2 else { gloves[hand] = general; return }
        pair[hand] = gloves[hand]
        weaponGloves?[weapon] = pair == gloves ? nil : pair
    }

    /// The turn an orb on `hand` is drawn with in third person.
    func bodyTurn(_ hand: Int) -> SIMD3<Float> {
        guard let t = bodyTurns, t.count == 2 else { return orbs[hand].turn }
        return t[hand]
    }

    mutating func clamp() {
        for i in 0..<2 {
            orbs[i].offset = simd_clamp(orbs[i].offset, SIMD3(repeating: -Self.offsetLimit), SIMD3(repeating: Self.offsetLimit))
            orbs[i].size = min(max(orbs[i].size, Self.orbSizeRange.lowerBound), Self.orbSizeRange.upperBound)
            bodyOrbs[i] = simd_clamp(bodyOrbs[i], SIMD3(repeating: -Self.offsetLimit), SIMD3(repeating: Self.offsetLimit))
            gloves[i].offset = simd_clamp(gloves[i].offset, SIMD3(repeating: -0.5), SIMD3(repeating: 0.5))
            gloves[i].scale = min(max(gloves[i].scale, Self.gloveScaleRange.lowerBound), Self.gloveScaleRange.upperBound)
        }
        if var all = weaponGloves {
            for (k, pair) in all {
                all[k] = pair.map { g in
                    var g = g
                    g.offset = simd_clamp(g.offset, SIMD3(repeating: -0.5), SIMD3(repeating: 0.5))
                    g.scale = min(max(g.scale, Self.gloveScaleRange.lowerBound), Self.gloveScaleRange.upperBound)
                    return g
                }
            }
            weaponGloves = all
        }
        reticleScale = min(max(reticleScale, Self.reticleScaleRange.lowerBound), Self.reticleScaleRange.upperBound)
        reticleDistance = min(max(reticleDistance, Self.reticleDistanceRange.lowerBound), Self.reticleDistanceRange.upperBound)
    }
}

/// The pieces an edit can pick up.
enum HUDPiece: Hashable, CustomStringConvertible {
    /// An orb at a hand, by the hand it sits at.
    case orb(hand: Int)
    /// An orb beside the body in third person, by the hand side it is on.
    case bodyOrb(hand: Int)
    case glove(hand: Int)
    case reticle

    var hand: Int? {
        switch self {
        case .orb(let h), .bodyOrb(let h), .glove(let h): return h
        case .reticle: return nil
        }
    }

    /// The same piece on the other side, for a mirrored edit.
    var partner: HUDPiece? {
        switch self {
        case .orb(let h): return .orb(hand: 1 - h)
        case .bodyOrb(let h): return .bodyOrb(hand: 1 - h)
        case .glove(let h): return .glove(hand: 1 - h)
        case .reticle: return nil
        }
    }

    /// Whether it rides on its own hand's controller, so only the other
    /// hand can move it.
    var ridesHand: Bool {
        switch self {
        case .orb, .glove: return true
        case .bodyOrb, .reticle: return false
        }
    }

    var description: String {
        let side = { (h: Int) in h == 0 ? "left" : "right" }
        switch self {
        case .orb(let h): return "\(side(h)) orb"
        case .bodyOrb(let h): return "\(side(h)) orb, third person"
        case .glove(let h): return "\(side(h)) glove"
        case .reticle: return "reticle"
        }
    }
}

/// The gizmo on the selected piece: which kind of handles it shows, and
/// the handles themselves.
enum GizmoMode: Int, CaseIterable, CustomStringConvertible {
    case move, rotate, scale

    var next: GizmoMode { GizmoMode(rawValue: (rawValue + 1) % 3)! }

    var description: String {
        switch self {
        case .move: return "Move"
        case .rotate: return "Rotate"
        case .scale: return "Scale"
        }
    }
}

enum GizmoHandle: Hashable {
    /// An arrow along axis 0, 1 or 2 (right, up, forward).
    case axis(Int)
    /// A ring about axis 0, 1 or 2 (pitch, yaw, roll).
    case ring(Int)
    case scale
}

/// Where a gizmo's handles are, and what a controller does to them: pure
/// geometry, so the offline test covers it.
struct Gizmo {
    var centre: SIMD3<Float>
    /// Unit axes, right, up and forward: the piece's own or the room's.
    var axes: simd_float3x3
    /// The piece's radius; the handles stand off it.
    var radius: Float
    var mode: GizmoMode

    var arrowStart: Float { radius * 1.15 }
    var arrowLength: Float { max(radius * 1.3, 0.05) }
    var ringRadius: Float { radius * 1.45 + 0.01 }
    var squareSize: Float { max(radius * 0.35, 0.015) }

    /// The scale handle: out along the right and up diagonal.
    var scalePoint: SIMD3<Float> {
        centre + simd_normalize(axes.columns.0 + axes.columns.1) * (radius * 1.5 + 0.01)
    }

    var handles: [GizmoHandle] {
        switch mode {
        case .move: return [.axis(0), .axis(1), .axis(2)]
        case .rotate: return [.ring(0), .ring(1), .ring(2)]
        case .scale: return [.scale]
        }
    }

    /// How far a point is from a handle, metres.
    func distance(from p: SIMD3<Float>, to handle: GizmoHandle) -> Float {
        switch handle {
        case .axis(let i):
            let a = axes[i]
            let s = centre + a * arrowStart
            let t = min(max(simd_dot(p - s, a), 0), arrowLength)
            return simd_length(p - (s + a * t))
        case .ring(let i):
            let a = axes[i]
            let v = p - centre
            let inPlane = v - a * simd_dot(v, a)
            let r = simd_length(inPlane)
            return simd_length(SIMD2(r - ringRadius, simd_dot(v, a)))
        case .scale:
            return max(simd_length(p - scalePoint) - squareSize * 0.5, 0)
        }
    }

    /// The handle a hand reaches (its grip within `reach`) or a ray
    /// points at, nearest first.
    func hit(point: SIMD3<Float>?, reach: Float = 0.03, rayOrigin: SIMD3<Float>?, rayDirection: SIMD3<Float>?) -> GizmoHandle? {
        var best: (GizmoHandle, Float)?
        for h in handles {
            var score = Float.greatestFiniteMagnitude
            if let point {
                let d = distance(from: point, to: h)
                if d < reach { score = d }
            }
            if score == .greatestFiniteMagnitude, let o = rayOrigin, let dir = rayDirection {
                // March the ray past the gizmo and take its nearest pass.
                let d = simd_normalize(dir)
                let along = simd_dot(centre - o, d)
                guard along > 0 else { continue }
                let span = ringRadius + arrowStart + arrowLength
                var nearest = Float.greatestFiniteMagnitude
                var t = max(along - span, 0)
                let step = span / 48
                while t <= along + span {
                    nearest = min(nearest, distance(from: o + d * t, to: h))
                    t += step
                }
                let tolerance = max(0.012, along * 0.02)
                if nearest < tolerance { score = 1 + nearest / along }
            }
            if score < (best?.1 ?? .greatestFiniteMagnitude) { best = (h, score) }
        }
        return best?.0
    }

    /// A move along axis `i`: how far the controller has moved along it,
    /// snapped to whole centimetres when `snap`.
    static func slide(from start: SIMD3<Float>, to now: SIMD3<Float>, axis: SIMD3<Float>, snap: Bool) -> Float {
        let d = simd_dot(now - start, axis)
        return snap ? (d / 0.01).rounded() * 0.01 : d
    }

    /// A turn about `axis` through `centre`: the hand's arc around it when
    /// the hand is far enough off the axis to make one, else the hand's
    /// own twist about it. Radians, snapped to 5 degrees when `snap`.
    static func turn(centre: SIMD3<Float>, axis: SIMD3<Float>, startPoint: SIMD3<Float>, point: SIMD3<Float>,
                     startTurn: simd_quatf, nowTurn: simd_quatf, snap: Bool) -> Float {
        func flat(_ v: SIMD3<Float>) -> SIMD3<Float> { v - axis * simd_dot(v, axis) }
        let v0 = flat(startPoint - centre), v1 = flat(point - centre)
        var angle: Float
        if simd_length(v0) > 0.03 && simd_length(v1) > 0.03 {
            angle = atan2(simd_dot(simd_cross(v0, v1), axis), simd_dot(v0, v1))
        } else {
            // The twist: the controller's turn since the grab, its part
            // about the axis.
            let q = nowTurn * startTurn.inverse
            let twist = simd_dot(q.imag, axis)
            angle = 2 * atan2(twist, q.real)
            if angle > .pi { angle -= 2 * .pi } else if angle < -.pi { angle += 2 * .pi }
        }
        if snap {
            let step: Float = 5 * .pi / 180
            angle = (angle / step).rounded() * step
        }
        return angle
    }

    /// A drag of the scale handle: the factor from how far the hand has
    /// moved out along the handle's direction, snapped to 5 per cent.
    static func scaleFactor(from start: SIMD3<Float>, to now: SIMD3<Float>, outward: SIMD3<Float>, radius: Float,
                            snap: Bool) -> Float {
        let d = simd_dot(now - start, simd_normalize(outward))
        var f = max(0.1, 1 + d / max(radius, 0.02))
        if snap { f = max(0.05, (f / 0.05).rounded() * 0.05) }
        return f
    }
}

/// The math the edit runs on: turning a piece's stored values into a
/// place in the room and a grabbed place back into stored values, both
/// ways through the same frames the compositor draws with.
enum HUDLayoutMath {

    // MARK: Turns

    /// The orb's turn as the compositor builds it: yaw (mirrored), then
    /// pitch, then roll (mirrored), about its own upright, across and
    /// facing axes.
    static func orbTurn(_ degrees: SIMD3<Float>, mirrored: Bool) -> simd_float3x3 {
        let sign: Float = mirrored ? -1 : 1
        let r = degrees * (.pi / 180)
        let yaw = simd_quatf(angle: r.y * sign, axis: [0, 1, 0])
        let pitch = simd_quatf(angle: -r.x, axis: [1, 0, 0])
        let roll = simd_quatf(angle: -r.z * sign, axis: [0, 0, 1])
        return simd_float3x3(yaw * pitch * roll)
    }

    /// Back from a turn matrix built as `orbTurn`: it is a yaw, pitch,
    /// roll product (Y then X then Z), read off its terms.
    static func orbDegrees(_ m: simd_float3x3, mirrored: Bool) -> SIMD3<Float> {
        let sign: Float = mirrored ? -1 : 1
        let b = asin(min(max(-m.columns.2.y, -1), 1))
        let a = atan2(m.columns.2.x, m.columns.2.z)
        let c = atan2(m.columns.0.y, m.columns.1.y)
        return SIMD3(-b, a * sign, -c * sign) * (180 / .pi)
    }

    /// The glove's turn as the compositor builds it: roll and yaw
    /// mirrored, applied Z, then Y, then X.
    static func gloveTurn(_ degrees: SIMD3<Float>, mirrored: Bool) -> simd_float3x3 {
        let sign: Float = mirrored ? -1 : 1
        let r = degrees * (.pi / 180)
        return simd_float3x3(
            simd_quatf(angle: r.z * sign, axis: [0, 0, 1]) * simd_quatf(angle: r.y * sign, axis: [0, 1, 0])
                * simd_quatf(angle: r.x, axis: [1, 0, 0])
        )
    }

    /// Back from a `gloveTurn` matrix, a Z, Y, X product.
    static func gloveDegrees(_ m: simd_float3x3, mirrored: Bool) -> SIMD3<Float> {
        let sign: Float = mirrored ? -1 : 1
        let b = asin(min(max(-m.columns.0.z, -1), 1))
        let a = atan2(m.columns.1.z, m.columns.2.z)
        let c = atan2(m.columns.0.y, m.columns.0.x)
        return SIMD3(a, b * sign, c * sign) * (180 / .pi)
    }

    /// The glove's fitting in its grip, without its scale.
    static func gloveFit(_ glove: HUDLayout.Glove, mirrored: Bool) -> simd_float4x4 {
        let sign: Float = mirrored ? -1 : 1
        let r = gloveTurn(glove.turn, mirrored: mirrored)
        return simd_float4x4(
            SIMD4(r.columns.0, 0), SIMD4(r.columns.1, 0), SIMD4(r.columns.2, 0),
            SIMD4(glove.offset.x * sign, glove.offset.y, glove.offset.z, 1)
        )
    }

    /// A fitting read back into a glove's values, keeping its scale.
    static func glove(fromFit fit: simd_float4x4, scale: Float, mirrored: Bool) -> HUDLayout.Glove {
        let sign: Float = mirrored ? -1 : 1
        let r = simd_float3x3(
            simd_normalize(SIMD3(fit.columns.0.x, fit.columns.0.y, fit.columns.0.z)),
            simd_normalize(SIMD3(fit.columns.1.x, fit.columns.1.y, fit.columns.1.z)),
            simd_normalize(SIMD3(fit.columns.2.x, fit.columns.2.y, fit.columns.2.z))
        )
        return HUDLayout.Glove(
            offset: SIMD3(fit.columns.3.x * sign, fit.columns.3.y, fit.columns.3.z),
            turn: gloveDegrees(r, mirrored: mirrored),
            scale: scale
        )
    }

    // MARK: The gun's frame

    /// The frame the game draws the first person gun in, in the room: the
    /// hand aim turns the view model to the aim turned down by the tilt
    /// about its own right axis, rolls it with the wrist, and moves it to
    /// the aim's place less the rest offset (forward, right, up in game
    /// units along the head's level heading, `RT_HANDAIM_REST`), as the
    /// runtime's handaim.c does. Gloves placed in this frame move with the
    /// gun as one piece. headYaw in radians, positive left.
    static func gunFrame(aim: simd_float4x4, tiltDegrees: Float, headYaw: Float, rest: SIMD3<Float>,
                         unitsPerMetre: Float) -> simd_float4x4 {
        let t = -tiltDegrees * .pi / 180
        var turn = matrix_identity_float4x4
        turn.columns.1 = [0, cos(t), sin(t), 0]
        turn.columns.2 = [0, -sin(t), cos(t), 0]
        var g = aim * turn
        let x = simd_normalize(SIMD3(g.columns.0.x, g.columns.0.y, g.columns.0.z))
        var y = SIMD3(g.columns.1.x, g.columns.1.y, g.columns.1.z)
        y = simd_normalize(y - simd_dot(y, x) * x)
        let z = simd_cross(x, y)
        let fwd = SIMD3<Float>(-sin(headYaw), 0, -cos(headYaw)), right = SIMD3<Float>(cos(headYaw), 0, -sin(headYaw))
        let lift = (rest.x * fwd + rest.y * right + rest.z * SIMD3<Float>(0, 1, 0)) / unitsPerMetre
        let p = SIMD3(aim.columns.3.x, aim.columns.3.y, aim.columns.3.z) - lift
        g = simd_float4x4(SIMD4(x, 0), SIMD4(y, 0), SIMD4(z, 0), SIMD4(p, 1))
        return g
    }

    /// A glove's place in one frame turned into another: the same glove in
    /// the room, measured from `to` instead of `from`.
    static func reanchor(_ glove: HUDLayout.Glove, mirrored: Bool, from: simd_float4x4, to: simd_float4x4) -> HUDLayout.Glove {
        let fit = to.inverse * from * gloveFit(glove, mirrored: mirrored)
        return self.glove(fromFit: fit, scale: glove.scale, mirrored: mirrored).keepingGrip(of: glove)
    }

    // MARK: Frames

    /// A frame in the room: where it is and which way its axes point.
    struct Frame {
        var origin: SIMD3<Float>
        var axes: simd_float3x3

        var matrix: simd_float4x4 {
            simd_float4x4(SIMD4(axes.columns.0, 0), SIMD4(axes.columns.1, 0), SIMD4(axes.columns.2, 0), SIMD4(origin, 1))
        }

        init(origin: SIMD3<Float>, axes: simd_float3x3) {
            self.origin = origin
            self.axes = axes
        }

        init(_ m: simd_float4x4) {
            origin = SIMD3(m.columns.3.x, m.columns.3.y, m.columns.3.z)
            axes = simd_float3x3(
                simd_normalize(SIMD3(m.columns.0.x, m.columns.0.y, m.columns.0.z)),
                simd_normalize(SIMD3(m.columns.1.x, m.columns.1.y, m.columns.1.z)),
                simd_normalize(SIMD3(m.columns.2.x, m.columns.2.y, m.columns.2.z))
            )
        }

        /// A point in the room, in this frame's own axes.
        func local(_ point: SIMD3<Float>) -> SIMD3<Float> { axes.transpose * (point - origin) }
    }

    /// A grab: the piece's place relative to the controller at the
    /// moment it was taken, so it follows the controller rigidly after,
    /// position and turn together.
    static func hold(piece: simd_float4x4, controller: simd_float4x4) -> simd_float4x4 {
        Frame(controller).matrix.inverse * Frame(piece).matrix
    }

    /// The stick across tilts a held piece about its own across axis at a
    /// steady rate, by turning it within the hold so the controller keeps
    /// carrying it: right leans an orb's top back (its pitch up, the
    /// porthole towards the eyes), left towards the player; a glove's
    /// pitch the same way.
    static func tilted(_ hold: simd_float4x4, piece: HUDPiece, stickX: Float, dt: Float) -> simd_float4x4 {
        guard abs(stickX) > 0.2, dt > 0 else { return hold }
        let degrees = stickX * HUDLayout.tiltRate * dt
        let angle: Float
        switch piece {
        case .orb, .bodyOrb: angle = -degrees * .pi / 180
        case .glove: angle = degrees * .pi / 180
        case .reticle: return hold
        }
        return hold * simd_float4x4(simd_quatf(angle: angle, axis: [1, 0, 0]))
    }

    /// Where a held piece is now.
    static func held(_ hold: simd_float4x4, controller: simd_float4x4) -> simd_float4x4 {
        Frame(controller).matrix * hold
    }

    /// An orb at a hand's values from where it was put: its centre in the
    /// anchor's axes, its turn against the facing frame the compositor
    /// gives it there.
    static func orb(at placed: simd_float4x4, anchor: Frame, facing: simd_float3x3, size: Float, mirrored: Bool) -> HUDLayout.Orb {
        let p = Frame(placed)
        let local = anchor.local(p.origin)
        return HUDLayout.Orb(
            offset: SIMD3(local.x * (mirrored ? -1 : 1), local.y, local.z),
            turn: orbDegrees(facing.transpose * p.axes, mirrored: mirrored),
            size: size
        )
    }

    /// The same for an orb beside the body: only its place.
    static func bodyOrb(at placed: simd_float4x4, body: Frame, mirrored: Bool) -> SIMD3<Float> {
        let local = body.local(Frame(placed).origin)
        return SIMD3(local.x * (mirrored ? -1 : 1), local.y, local.z)
    }

    /// The upright frame the compositor turns an orb from: up the room's,
    /// facing back towards the head across the floor.
    static func facingFrame(facing: SIMD3<Float>) -> simd_float3x3 {
        let up = SIMD3<Float>(0, 1, 0)
        let x = simd_normalize(simd_cross(up, facing))
        return simd_float3x3(x, up, facing)
    }

    /// The nearest point between a ray and a point: how far along the
    /// ray, and how far off it.
    static func rayDistance(origin: SIMD3<Float>, direction: SIMD3<Float>, to point: SIMD3<Float>) -> (along: Float, off: Float) {
        let d = simd_normalize(direction)
        let along = simd_dot(point - origin, d)
        let off = simd_length(point - (origin + d * max(along, 0)))
        return (along, off)
    }
}
