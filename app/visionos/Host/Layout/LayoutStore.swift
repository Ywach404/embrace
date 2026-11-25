import Foundation
import simd

/// The live layout: what the compositor draws with and the edit changes.
/// It starts from the built in defaults, then the player's saved layout,
/// then any `RT_` setting given at launch, which wins for that run. The
/// saved layout is a JSON file in the app's own Application Support, so
/// it survives a reinstall of the same build and never leaves the device.
/// Any thread; the compositor reads it once a frame.
final class LayoutStore: @unchecked Sendable {
    nonisolated static let shared = LayoutStore()

    private let lock = NSLock()
    private var value: HUDLayout
    private var weaponNow = ""

    /// The weapon in hand as the runtime names it (rt_host_weapon), "" for
    /// none: which weapon's glove places the compositor draws and the edit
    /// changes. Set by the compositor once a frame.
    var weapon: String {
        get { lock.lock(); defer { lock.unlock() }; return weaponNow }
        set { lock.lock(); weaponNow = newValue; lock.unlock() }
    }

    /// The layout as built, before anything saved or set.
    static let defaults: HUDLayout = {
        let tilt = SpatialInput.orbTilt
        let size = SpatialInput.orbSize
        let orb = { (hand: Int) in
            HUDLayout.Orb(offset: SIMD3(SpatialInput.orbOffset, 0.07, 0), turn: tilt[hand], size: size)
        }
        let body = SIMD3<Float>(SpatialInput.orbThirdPersonSide, -SpatialInput.orbThirdPersonHeight,
                                SpatialInput.orbThirdPersonForward)
        let glove = HUDLayout.Glove(offset: SpatialInput.handsOffset, turn: SpatialInput.handsRotation,
                                    scale: SpatialInput.handsScale)
        return HUDLayout(orbs: [orb(0), orb(1)], bodyOrbs: [body, body], bodyTurns: [tilt[0], tilt[1]],
                         gloves: [glove, glove],
                         reticleDistance: 8, reticleScale: 1)
    }()

    private static var fileURL: URL? {
        guard let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first else {
            return nil
        }
        return base.appendingPathComponent("layout.json")
    }

    private init() {
        var layout = Self.defaults
        var source = "the defaults"
        if let url = Self.fileURL, let data = try? Data(contentsOf: url),
           let saved = try? JSONDecoder().decode(HUDLayout.self, from: data),
           saved.orbs.count == 2, saved.bodyOrbs.count == 2, saved.gloves.count == 2 {
            layout = saved
            source = "the saved layout"
        }
        // A setting given at launch is the player asking for that value
        // this run, so it wins over what was saved.
        let env = ProcessInfo.processInfo.environment
        let d = Self.defaults
        if env["RT_ORB_OFFSET"] != nil { for i in 0..<2 { layout.orbs[i].offset.x = d.orbs[i].offset.x } }
        if env["RT_ORB_SIZE"] != nil { for i in 0..<2 { layout.orbs[i].size = d.orbs[i].size } }
        if env["RT_ORB_PITCH"] != nil || env["RT_ORB_YAW"] != nil || env["RT_ORB_ROLL"] != nil
            || env["RT_ORB_TILT_SPLIT"] != nil {
            for i in 0..<2 { layout.orbs[i].turn = d.orbs[i].turn }
            layout.bodyTurns = d.bodyTurns
        }
        if env["RT_ORB_TP_SIDE"] != nil || env["RT_ORB_TP_HEIGHT"] != nil || env["RT_ORB_TP_FORWARD"] != nil {
            layout.bodyOrbs = d.bodyOrbs
        }
        if env["RT_HANDS_OFFSET"] != nil { for i in 0..<2 { layout.gloves[i].offset = d.gloves[i].offset } }
        if env["RT_HANDS_ROT"] != nil { for i in 0..<2 { layout.gloves[i].turn = d.gloves[i].turn } }
        if env["RT_HANDS_SCALE"] != nil { for i in 0..<2 { layout.gloves[i].scale = d.gloves[i].scale } }
        layout.clamp()
        value = layout
        rt_host_log("layout: from \(source); \(Self.describe(layout))")
    }

    var current: HUDLayout {
        lock.lock()
        defer { lock.unlock() }
        return value
    }

    func update(_ change: (inout HUDLayout) -> Void) {
        lock.lock()
        change(&value)
        value.clamp()
        lock.unlock()
    }

    func replace(_ layout: HUDLayout) {
        update { $0 = layout }
    }

    /// Writes the layout out and logs it whole, so the values can be read
    /// off the run log and made the defaults.
    func save() {
        let layout = current
        guard let url = Self.fileURL else { return }
        do {
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(layout).write(to: url, options: .atomic)
            rt_host_log("layout: saved; \(Self.describe(layout))")
        } catch {
            rt_host_log("layout: not saved (\(error.localizedDescription))")
        }
    }

    static func describe(_ l: HUDLayout) -> String {
        func cm(_ v: SIMD3<Float>) -> String {
            String(format: "%.1f, %.1f, %.1f cm", v.x * 100, v.y * 100, v.z * 100)
        }
        func deg(_ v: SIMD3<Float>) -> String {
            String(format: "pitch %.0f yaw %.0f roll %.0f", v.x, v.y, v.z)
        }
        var parts: [String] = []
        for (i, side) in ["left", "right"].enumerated() {
            let o = l.orbs[i], g = l.gloves[i]
            parts.append("\(side) orb \(cm(o.offset)), \(deg(o.turn)), \(String(format: "%.1f", o.size * 100)) cm across")
            parts.append("\(side) orb in third person \(cm(l.bodyOrbs[i])), \(deg(l.bodyTurn(i)))")
            parts.append("\(side) glove \(cm(g.offset)), \(deg(g.turn)), scale \(String(format: "%.2f", g.scale))")
        }
        for (w, pair) in (l.weaponGloves ?? [:]).sorted(by: { $0.key < $1.key }) where pair.count == 2 {
            for (i, side) in ["left", "right"].enumerated() {
                let g = pair[i]
                parts.append("\(side) glove on \(w) \(cm(g.offset)), \(deg(g.turn)), scale \(String(format: "%.2f", g.scale))")
            }
        }
        parts.append(String(format: "reticle %.1f m, scale %.2f", l.reticleDistance, l.reticleScale))
        return parts.joined(separator: "; ")
    }
}
