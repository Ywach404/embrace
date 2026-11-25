import Foundation
import simd

/// The gloves on the game's own hands. In first person the game's view
/// model (gun and arms in one skinned model) is posed by its own
/// animations, reloads, recoil and swaps, on top of the hand aim that moves
/// it to the controller; the runtime reads its hand bones and the backend
/// carries them into each shown frame (`vm_hand_*`) as the aim point is.
/// With `RT_HANDS_ATTACH=1` the gun hand's glove sits on the game's right
/// hand bone while a weapon is out, so it moves with the gun, and on a
/// rifle the other glove takes the game's left hand when the real hand is
/// near it.
///
/// The glove's place relative to the bone is the fitting. By default it
/// is taken from the hands themselves: once the bone has held still in the
/// grip for half a second, the glove's place on the controller then is
/// stored relative to the bone, so the glove starts where it was and from
/// then on follows the game's animation. Every weapon is held by the same
/// arms model (A.'s gun run, 2026-11-25), so one fitting serves them all;
/// it is kept across launches, and `RT_HANDS_ATTACH_REFIT=1` takes a new
/// one. `RT_HANDS_ATTACH_FIT=x,y,z,pitch,yaw,roll` (cm, degrees, in the
/// bone's frame, mirrored for the left hand) sets it by hand instead.
/// Used only from the compositor's render thread, as ScreenCompositor is.
final class HandAttach: @unchecked Sendable {
    static let shared = HandAttach()

    let enabled: Bool
    private let manualFit: simd_float4x4?

    /// The game's hand bones this frame in the origin's space: right, left,
    /// and the drawn palms' middles, with the frame's eyes and turn for the
    /// log's eye space.
    private var bones: [simd_float4x4?] = [nil, nil]
    private var palms: [SIMD3<Float>?] = [nil, nil]
    private var eyeMid = SIMD4<Float>(0, 0, 0, 1)
    private var eyeTurn = matrix_identity_float4x4
    private var frameTime: TimeInterval = 0
    private var frameDt: Float = 0
    /// What the gun hand holds this frame: 0 nothing (or a menu, or not
    /// full view), 1 a pistol, 2 a rifle, 3 a melee weapon.
    var weaponClass = 0
    /// Per glove side (HandPose's: 0 left, 1 right): the fitting found,
    /// the stillness it is waiting on, and the off hand's blend toward the
    /// game's hand.
    private var fits: [simd_float4x4?] = [nil, nil]
    private var still: [(since: TimeInterval, last: simd_float4x4?)] = [(0, nil), (0, nil)]
    private var blend: [Float] = [0, 0]
    private var logged = Set<String>()
    private var lastLog: TimeInterval = 0

    private init() {
        let env = ProcessInfo.processInfo.environment
        enabled = (env["RT_HANDS_ATTACH"].flatMap(Int.init) ?? 0) != 0
        manualFit = Self.parseFit(env["RT_HANDS_ATTACH_FIT"])
        if (env["RT_HANDS_ATTACH_REFIT"].flatMap(Int.init) ?? 0) != 0 {
            UserDefaults.standard.removeObject(forKey: Self.savedKey)
        } else if manualFit == nil, let saved = UserDefaults.standard.array(forKey: Self.savedKey) as? [Float], saved.count == 16 {
            let m = simd_float4x4(SIMD4(saved[0], saved[1], saved[2], saved[3]), SIMD4(saved[4], saved[5], saved[6], saved[7]),
                                  SIMD4(saved[8], saved[9], saved[10], saved[11]), SIMD4(saved[12], saved[13], saved[14], saved[15]))
            fits[1] = m
            ScreenCompositor.log("hands attach: the saved fitting for the game's arms, \(Self.describe(m))")
        }
    }

    /// Where the gun hand's fitting is kept between launches.
    static let savedKey = "handsAttachFit2"

    private func save(_ m: simd_float4x4) {
        let v = (0..<4).flatMap { c in (0..<4).map { r in m[c][r] } }
        UserDefaults.standard.set(v, forKey: Self.savedKey)
    }

    /// "x,y,z,pitch,yaw,roll", centimetres and degrees; nil for "auto" or
    /// nothing given.
    static func parseFit(_ text: String?) -> simd_float4x4? {
        guard let text, text.lowercased() != "auto" else { return nil }
        let v = text.split(separator: ",").compactMap { Float($0.trimmingCharacters(in: .whitespaces)) }
        guard v.count == 6 else { return nil }
        return fitMatrix(v)
    }

    static func fitMatrix(_ v: [Float]) -> simd_float4x4 {
        let d = Float.pi / 180
        let rx = simd_quatf(angle: v[3] * d, axis: SIMD3(1, 0, 0))
        let ry = simd_quatf(angle: v[4] * d, axis: SIMD3(0, 1, 0))
        let rz = simd_quatf(angle: v[5] * d, axis: SIMD3(0, 0, 1))
        var m = simd_float4x4(ry * rx * rz)
        m.columns.3 = SIMD4(v[0] / 100, v[1] / 100, v[2] / 100, 1)
        return m
    }

    /// A hand's frame from its three points (origin, along x, along y):
    /// x and y made orthonormal, z their cross product. nil for points
    /// too close to give axes.
    static func frame(origin p0: SIMD3<Float>, alongX p1: SIMD3<Float>, alongY p2: SIMD3<Float>) -> simd_float4x4? {
        let ax = p1 - p0
        guard simd_length(ax) > 1e-4 else { return nil }
        let x = simd_normalize(ax)
        var ay = p2 - p0
        ay -= simd_dot(ay, x) * x
        guard simd_length(ay) > 1e-4 else { return nil }
        let y = simd_normalize(ay)
        let z = simd_cross(x, y)
        return simd_float4x4(SIMD4(x, 0), SIMD4(y, 0), SIMD4(z, 0), SIMD4(p0, 1))
    }

    /// A point the backend gave as tangents and metres off the drawn
    /// forward, placed as the third person aim's is: from the eyes'
    /// midpoint, turned by the orientation the frame was drawn with.
    static func place(tan: (Float, Float), metres: Float, mid: SIMD4<Float>, drawn: simd_float4x4) -> SIMD3<Float> {
        let p = mid + drawn * SIMD4(tan.0 * metres, tan.1 * metres, -metres, 0)
        return SIMD3(p.x, p.y, p.z)
    }

    /// Once a frame: the game's hands from the shown frame.
    func update(shown: metal_d3d9_shown_frame, mid: SIMD4<Float>, drawn: simd_float4x4, now: TimeInterval) {
        guard enabled, now != frameTime else { return }
        frameDt = frameTime == 0 ? 0 : Float(min(max(now - frameTime, 0), 0.1))
        frameTime = now
        eyeMid = mid
        eyeTurn = drawn
        var tan = shown.vm_hand_tan, m = shown.vm_hand_m, on = shown.vm_hand_shown
        var ptan = shown.vm_palm_tan, pm = shown.vm_palm_m
        let palmT = withUnsafeBytes(of: &ptan) { Array($0.bindMemory(to: Float.self)) }
        let palmM = withUnsafeBytes(of: &pm) { Array($0.bindMemory(to: Float.self)) }
        let t = withUnsafeBytes(of: &tan) { Array($0.bindMemory(to: Float.self)) }
        let metres = withUnsafeBytes(of: &m) { Array($0.bindMemory(to: Float.self)) }
        let shownMask = withUnsafeBytes(of: &on) { Array($0.bindMemory(to: Int32.self)) }
        for k in 0..<2 {
            guard shown.fullview != 0, shown.pose_valid != 0, shownMask[k] != 0 else { bones[k] = nil; palms[k] = nil; continue }
            palms[k] = palmM[k] > 0 ? Self.place(tan: (palmT[2 * k], palmT[2 * k + 1]), metres: palmM[k], mid: mid, drawn: drawn) : nil
            let pts = (0..<3).map { j in
                Self.place(tan: (t[2 * (3 * k + j)], t[2 * (3 * k + j) + 1]), metres: metres[3 * k + j], mid: mid, drawn: drawn)
            }
            bones[k] = Self.frame(origin: pts[0], alongX: pts[1], alongY: pts[2])
        }
    }

    /// Where a glove goes instead of its place on the controller (`placed`,
    /// the grip with its fitting), or nil to leave it there. `side` is
    /// HandPose's (0 left, 1 right) and `gunSide` the aiming hand's.
    func attached(side: Int, gunSide: Int, placed: simd_float4x4, now: TimeInterval) -> simd_float4x4? {
        guard enabled else { return nil }
        let dt = frameDt
        let gunHand = side == gunSide
        // The game's right hand holds the weapon whichever hand aims.
        let bone = bones[gunHand ? 0 : 1]
        guard weaponClass != 0, let bone else {
            still[side] = (0, nil)
            blend[side] = 0
            return nil
        }
        // One fitting for the arms, whatever the weapon.
        if fits[side] == nil {
            if let manual = manualFit {
                fits[side] = side == 0 ? Self.mirrored(manual) : manual
            } else if gunHand {
                // The bone as seen from the grip; once it holds still for
                // half a second, the glove's place there is the fitting.
                // Only from a hand near the controller: the hand aim keeps the
                // game's drawn hand at the real one, so a bone farther than
                // 25 cm is a bad read, not a place to fit to.
                let inGrip = placed.inverse * bone
                let near = simd_distance(placed.columns.3, bone.columns.3) < 0.25
                if !near {
                    still[side] = (0, nil)
                    note("far-\(Int(now) / 5)", String(format: "hands attach: no fitting yet, the game's hand is %.0f cm from the controller", simd_distance(placed.columns.3, bone.columns.3) * 100))
                } else if let last = still[side].last, simd_distance(inGrip.columns.3, last.columns.3) < 0.01 {
                    if now - still[side].since > 0.5 {
                        let fit = bone.inverse * placed
                        fits[side] = fit
                        if side == 1 { save(fit) }
                        note("fit-\(side)", "hands attach: the \(side == 1 ? "right" : "left") glove fitted to the game's arms (weapon class \(weaponClass)), \(Self.describe(fit))")
                    }
                } else {
                    still[side] = (now, inGrip)
                }
                if fits[side] == nil { return nil }
            } else if let gun = fits[gunSide] {
                fits[side] = Self.mirrored(gun)
            } else {
                return nil
            }
        }
        guard let fit = fits[side] else { return nil }
        let onBone = bone * fit
        logEverySecond(side: side, gunHand: gunHand, placed: placed, onBone: onBone, now: now)
        if gunHand { return onBone }
        // The off hand: only a rifle's, and only while the real hand is
        // within 15 cm of the game's, eased in and out.
        let near = weaponClass == 2 && simd_distance(placed.columns.3, onBone.columns.3) < 0.15
        let k = 1 - exp(-dt / 0.1)
        blend[side] += ((near ? 1 : 0) - blend[side]) * k
        guard blend[side] > 0.01 else { return nil }
        return Self.mix(placed, onBone, blend[side])
    }

    /// A point in the frame's eye space: from the eyes' midpoint, turned
    /// back by the orientation the frame was drawn with (x right, y up, z
    /// toward the viewer), metres.
    private func eyeSpace(_ p: SIMD3<Float>) -> SIMD3<Float> {
        let v = eyeTurn.inverse * SIMD4(p.x - eyeMid.x, p.y - eyeMid.y, p.z - eyeMid.z, 0)
        return SIMD3(v.x, v.y, v.z)
    }

    /// A fitting for the other hand: x across the hand is flipped, and the
    /// turns about y and z with it.
    static func mirrored(_ m: simd_float4x4) -> simd_float4x4 {
        let s = simd_float4x4(diagonal: SIMD4(-1, 1, 1, 1))
        return s * m * s
    }

    static func mix(_ a: simd_float4x4, _ b: simd_float4x4, _ t: Float) -> simd_float4x4 {
        let qa = simd_quatf(simd_float3x3(SIMD3(a.columns.0.x, a.columns.0.y, a.columns.0.z),
                                          SIMD3(a.columns.1.x, a.columns.1.y, a.columns.1.z),
                                          SIMD3(a.columns.2.x, a.columns.2.y, a.columns.2.z)))
        let qb = simd_quatf(simd_float3x3(SIMD3(b.columns.0.x, b.columns.0.y, b.columns.0.z),
                                          SIMD3(b.columns.1.x, b.columns.1.y, b.columns.1.z),
                                          SIMD3(b.columns.2.x, b.columns.2.y, b.columns.2.z)))
        var m = simd_float4x4(simd_slerp(qa, qb, t))
        m.columns.3 = a.columns.3 + (b.columns.3 - a.columns.3) * t
        m.columns.3.w = 1
        return m
    }

    static func describe(_ m: simd_float4x4) -> String {
        let p = m.columns.3 * 100
        let q = simd_quatf(simd_float3x3(SIMD3(m.columns.0.x, m.columns.0.y, m.columns.0.z),
                                         SIMD3(m.columns.1.x, m.columns.1.y, m.columns.1.z),
                                         SIMD3(m.columns.2.x, m.columns.2.y, m.columns.2.z)))
        return String(format: "offset %.1f %.1f %.1f cm, turn %.0f degrees about %.2f %.2f %.2f",
                      p.x, p.y, p.z, q.angle * 180 / .pi, q.axis.x, q.axis.y, q.axis.z)
    }

    private func note(_ key: String, _ text: String) {
        guard logged.insert(key).inserted else { return }
        ScreenCompositor.log(text)
    }

    /// Once a second: how far the game's hand sits from the controller, so
    /// a run shows the fitting holding through reloads and swaps.
    private func logEverySecond(side: Int, gunHand: Bool, placed: simd_float4x4, onBone: simd_float4x4, now: TimeInterval) {
        guard gunHand, now - lastLog >= 1, logged.count < 400 else { return }
        lastLog = now
        logged.insert("tick-\(Int(now))")
        let d = simd_distance(placed.columns.3, onBone.columns.3) * 100
        let wrist = eyeSpace(SIMD3(onBone.columns.3.x, onBone.columns.3.y, onBone.columns.3.z)) * 100
        var palmText = "no drawn palm"
        if let palm = palms[0] {
            let pe = eyeSpace(palm) * 100
            palmText = String(format: "drawn right palm at %.1f %.1f %.1f cm, %.1f cm from the glove's wrist", pe.x, pe.y, pe.z, simd_distance(pe, wrist))
        }
        ScreenCompositor.log(String(format: "hands attach: the gun hand's glove on the game's hand, %.1f cm from its place on the controller; in eye space the glove's wrist at %.1f %.1f %.1f cm, ", d, wrist.x, wrist.y, wrist.z) + palmText)
    }
}
