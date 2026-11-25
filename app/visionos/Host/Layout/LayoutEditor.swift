import Foundation
import QuartzCore
import simd

/// What the edit reads from each controller half, set by
/// `ControllerWatcher` on its 120 Hz tick: the stick's up and down, its
/// click, the two face buttons under the thumb (Cross or Square the
/// lower, Circle or Triangle the upper) and the half's menu button
/// (Options on the right). The trigger comes from `HandControls`, which
/// already has it. Read straight from the controllers, so the edit sees
/// them while the game's pad is paused. Any thread.
struct EditButtons: Sendable {
    var stickX: Float = 0
    var stickY: Float = 0
    var stickClick = false
    var lower = false
    var upper = false
    var menu = false
}

final class EditInput: @unchecked Sendable {
    nonisolated static let shared = EditInput()
    private let lock = NSLock()
    private var sides = [EditButtons(), EditButtons()]

    func set(side: Int, _ buttons: EditButtons) {
        guard side == 0 || side == 1 else { return }
        lock.lock()
        sides[side] = buttons
        lock.unlock()
    }

    func get() -> [EditButtons] {
        lock.lock()
        defer { lock.unlock() }
        return sides
    }
}

/// The in game layout edit. Both stick clicks held together for a second
/// and a half open it and, held again, close it and save; `RT_EDIT=1`
/// opens it at launch. Two ways out besides: a tap of Options closes and
/// saves, and Circle or Triangle held for a second with nothing in that
/// hand closes and puts back the layout it opened with. While it is open
/// the game takes no input from the controllers (`rt_host_pad_pause`).
///
/// Open, each controller can pick a piece up: reach to it, or point at
/// it from further off, and hold the trigger. It then follows that
/// controller, its place and its turn together, as if held, and is put
/// down where the trigger lets go. A piece riding on one hand (its orb,
/// its glove) is picked up with the other. While holding, the stick up
/// and down grows and shrinks it, across tilts it (right leans its top
/// back, left towards the player), the lower face button turns mirroring
/// on and off (on, the other side's piece copies every change), and the
/// upper face button puts it back as built.
///
/// The compositor calls `step` once a frame with where each piece is
/// drawn and the controllers' poses, draws from `LayoutStore`, and draws
/// `overlay` on top: a ring on each piece in reach, its values beside
/// the one held, and a strip of the controls ahead.
final class LayoutEditor: @unchecked Sendable {
    nonisolated static let shared = LayoutEditor()

    /// Where a piece is drawn this frame, from the compositor.
    struct Placed {
        /// The piece's place, unscaled: for an orb its centre and turned
        /// axes, for a glove its grip times its fitting, for the reticle
        /// its centre facing the head.
        var world: simd_float4x4
        /// Its size for picking and for the ring, metres.
        var radius: Float
        /// The frame its stored offset is in: the hand's grip in the
        /// head's axes for an orb, the body for one in third person, the
        /// grip for a glove, the aim ray for the reticle.
        var anchor: HUDLayoutMath.Frame
        /// For an orb at a hand or beside the body, the upright facing
        /// frame its turn is from.
        var facing: simd_float3x3 = matrix_identity_float3x3
    }

    /// A controller this frame: its grip and aim poses in the room, nil
    /// when not tracked.
    struct Controller {
        var grip: simd_float4x4
        var aim: simd_float4x4
    }

    /// What to draw over the scene while editing.
    struct Overlay {
        struct Mark {
            var piece: HUDPiece
            var centre: SIMD3<Float>
            var radius: Float
            var held: Bool
            var text: String?
        }
        /// The gizmo on the selected piece: its handles, the one either
        /// hand is over and the one being dragged.
        struct GizmoMark {
            var gizmo: Gizmo
            var hovered: Set<GizmoHandle>
            var active: GizmoHandle?
        }
        var marks: [Mark] = []
        var gizmo: GizmoMark?
        var banner: String?
    }

    private struct Grab {
        var piece: HUDPiece
        /// The gizmo handle taken, nil for the piece itself (free).
        var handle: GizmoHandle? = nil
        /// For a handle: the piece's place, the gizmo's axes, the grip's
        /// turn and the piece's size when it was taken.
        var startWorld = matrix_identity_float4x4
        var startAxes = matrix_identity_float3x3
        var startTurn = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)
        var startSize: Float = 1
        var hold: simd_float4x4
        var startGrip: SIMD3<Float>
        var startDistance: Float
        var mirror: Bool
        var lowerWas = true
        var upperWas = true
    }

    private let lock = NSLock()
    private(set) var active = false
    private var chordSince: Double?
    private var chordFired = false
    private var grabs: [Grab?] = [nil, nil]
    private var triggerWas = [false, false]
    private var lastTime: Double = 0
    private var overlayValue = Overlay()
    private var mirrorOrbs = true
    private var mirrorGloves = true
    /// The layout as it was when the edit opened, for leaving unsaved.
    private var openedWith: HUDLayout?
    private var menuSince: [Double?] = [nil, nil]
    private var upperSince: [Double?] = [nil, nil]
    /// Circle or Triangle let go since it last put a piece back, so the
    /// same press cannot go on to close the edit unsaved.
    private var upperFree = [true, true]
    private var chordLogged = false
    /// The gizmo: its mode, its space and the piece it is on.
    private var mode = GizmoMode.move
    private var worldSpace = false
    private var selected: HUDPiece?
    private var hovered: [GizmoHandle?] = [nil, nil]
    private var lowerWasFree = [false, false]
    private var clickSince: [Double?] = [nil, nil]
    private var clickSpoilt = [false, false]

    static let chordHold: Double = 1.5
    static let menuTap: Double = 0.6
    static let discardHold: Double = 1.0

    private init() {
        if ProcessInfo.processInfo.environment["RT_EDIT"] == "1" {
            active = true
            openedWith = LayoutStore.shared.current
            rt_host_pad_pause(1)
            rt_host_log("layout edit: open at launch (RT_EDIT=1)")
        }
    }

    var isActive: Bool {
        lock.lock()
        defer { lock.unlock() }
        return active
    }

    var overlay: Overlay {
        lock.lock()
        defer { lock.unlock() }
        return overlayValue
    }

    /// Whether a stick click chord is in progress or just finished, so the
    /// stereo toggle on the same chord's quick press can stand aside.
    var chordHeldLong: Bool {
        lock.lock()
        defer { lock.unlock() }
        return chordFired
    }

    /// Once a compositor frame. `pieces` holds every piece drawn this
    /// frame; `controllers` each hand's poses (index 0 left, 1 right).
    func step(pieces: [HUDPiece: Placed], controllers: [Controller?], head: simd_float4x4) {
        lock.lock()
        defer { lock.unlock() }
        let now = CACurrentMediaTime()
        let dt = Float(lastTime == 0 ? 0 : min(max(now - lastTime, 0), 0.1))
        lastTime = now
        let buttons = EditInput.shared.get()

        // Both stick clicks held: open or close once the hold is long
        // enough, once per hold.
        if buttons[0].stickClick && buttons[1].stickClick {
            let since = chordSince ?? now
            chordSince = since
            if active, !chordLogged {
                chordLogged = true
                rt_host_log("layout edit: both stick clicks down; closing after \(Self.chordHold) s held")
            }
            if !chordFired, now - since >= Self.chordHold {
                chordFired = true
                setActive(!active)
            }
        } else {
            chordSince = nil
            chordFired = false
            chordLogged = false
        }
        guard active else {
            overlayValue = Overlay()
            menuSince = [nil, nil]
            upperSince = [nil, nil]
            return
        }

        // The ways out: Options tapped closes and saves; Circle or
        // Triangle held with nothing in that hand closes unsaved.
        for side in 0..<2 {
            let b = buttons[side]
            if b.menu {
                if menuSince[side] == nil { menuSince[side] = now }
            } else if let since = menuSince[side] {
                menuSince[side] = nil
                if now - since <= Self.menuTap {
                    rt_host_log("layout edit: Options tapped")
                    setActive(false)
                    return
                }
            }
            if !b.upper { upperFree[side] = true }
            if b.upper && grabs[side] == nil && upperFree[side] {
                let since = upperSince[side] ?? now
                upperSince[side] = since
                if now - since >= Self.discardHold {
                    rt_host_log("layout edit: Circle or Triangle held; leaving without saving")
                    if let openedWith { LayoutStore.shared.replace(openedWith) }
                    setActive(false, save: false)
                    return
                }
            } else {
                upperSince[side] = nil
            }
        }

        // With nothing in that hand: Cross or Square cycles the gizmo's
        // mode; one stick click tapped alone swaps local and room axes.
        for side in 0..<2 {
            let b = buttons[side]
            if grabs[side] == nil, b.lower, !lowerWasFree[side] {
                mode = mode.next
                rt_host_log("layout edit: gizmo mode \(mode)")
            }
            // Any press counts as seen, held or not, so the press that
            // flipped mirroring cannot go on to change the mode.
            lowerWasFree[side] = b.lower
            if b.stickClick {
                if clickSince[side] == nil { clickSince[side] = now; clickSpoilt[side] = false }
                if buttons[1 - side].stickClick { clickSpoilt[side] = true }
            } else if let since = clickSince[side] {
                clickSince[side] = nil
                if !clickSpoilt[side], now - since < 0.4 {
                    worldSpace.toggle()
                    rt_host_log("layout edit: gizmo axes \(worldSpace ? "the room's" : "the piece's own")")
                }
            }
        }

        let layout = LayoutStore.shared.current
        let gizmo = selected.flatMap { piece in pieces[piece].flatMap { gizmoFor(piece, $0) } }
        for side in 0..<2 {
            let (controls, _) = HandControls.shared.state(side: side)
            let down = controls.trigger > (triggerWas[side] ? 0.3 : 0.6)
            defer { triggerWas[side] = down }
            guard let controller = controllers[side] else {
                if grabs[side] != nil { release(side) }
                continue
            }
            if var grab = grabs[side] {
                guard down else {
                    release(side)
                    continue
                }
                let b = buttons[side]
                // The lower face button flips mirroring for this kind of
                // piece; the upper puts the piece back as built.
                if b.lower && !grab.lowerWas, grab.piece.partner != nil {
                    grab.mirror.toggle()
                    if case .glove = grab.piece { mirrorGloves = grab.mirror } else { mirrorOrbs = grab.mirror }
                    rt_host_log("layout edit: mirroring \(grab.mirror ? "on" : "off") for the \(grab.piece)")
                }
                if b.upper && !grab.upperWas {
                    upperFree[side] = false
                    reset(grab.piece, mirror: grab.mirror)
                    grabs[side] = nil
                    continue
                }
                grab.lowerWas = b.lower
                grab.upperWas = b.upper
                var sizeStick = b.stickY
                if grab.handle == nil {
                    if case .glove(let h) = grab.piece, mode == .scale {
                        // In Scale mode a held glove's stick sets its grip:
                        // across how far the fingers close, up and down the
                        // thumb; the square still sizes it.
                        adjustGrip(h, curl: b.stickX, thumb: b.stickY, dt: dt, mirror: grab.mirror)
                        sizeStick = 0
                    } else {
                        grab.hold = HUDLayoutMath.tilted(grab.hold, piece: grab.piece, stickX: b.stickX, dt: dt)
                    }
                }
                grabs[side] = grab
                let snap = HandControls.shared.state(side: 1 - side).0.grip > 0.5
                move(grab, controller: controller, stickY: sizeStick, dt: dt, pieces: pieces, head: head, layout: layout, snap: snap)
            } else if down && !triggerWas[side], let gizmo, let piece = selected, let placed = pieces[piece],
                      !(piece.ridesHand && piece.hand == side),
                      let handle = gizmo.hit(point: HUDLayoutMath.Frame(controller.grip).origin,
                                             rayOrigin: HUDLayoutMath.Frame(controller.aim).origin,
                                             rayDirection: -HUDLayoutMath.Frame(controller.aim).axes.columns.2) {
                let grip = HUDLayoutMath.Frame(controller.grip)
                let mirror: Bool
                if case .glove = piece { mirror = mirrorGloves } else { mirror = mirrorOrbs }
                grabs[side] = Grab(
                    piece: piece, handle: handle, startWorld: placed.world, startAxes: gizmo.axes,
                    startTurn: simd_quatf(grip.axes), startSize: size(of: piece, in: layout),
                    hold: HUDLayoutMath.hold(piece: placed.world, controller: controller.grip),
                    startGrip: grip.origin, startDistance: layout.reticleDistance, mirror: mirror,
                    lowerWas: buttons[side].lower, upperWas: buttons[side].upper
                )
                rt_host_log("layout edit: took the \(piece)'s \(handle) handle with the \(side == 0 ? "left" : "right") hand")
            } else if down && !triggerWas[side] {
                if let (piece, placed) = pick(side: side, controller: controller, pieces: pieces) {
                    selected = piece
                    let grip = HUDLayoutMath.Frame(controller.grip)
                    let mirror: Bool
                    if case .glove = piece { mirror = mirrorGloves } else { mirror = mirrorOrbs }
                    grabs[side] = Grab(
                        piece: piece,
                        hold: HUDLayoutMath.hold(piece: placed.world, controller: controller.grip),
                        startGrip: grip.origin,
                        startDistance: layout.reticleDistance,
                        mirror: mirror,
                        lowerWas: buttons[side].lower,
                        upperWas: buttons[side].upper
                    )
                    rt_host_log("layout edit: picked up the \(piece) with the \(side == 0 ? "left" : "right") hand")
                }
            }
        }

        // What each free hand is over: a handle of the selected piece's
        // gizmo, else a piece, which becomes the selected one.
        for side in 0..<2 {
            hovered[side] = nil
            guard grabs[side] == nil, let controller = controllers[side] else { continue }
            let aim = HUDLayoutMath.Frame(controller.aim)
            if let gizmo, let piece = selected, !(piece.ridesHand && piece.hand == side),
               let h = gizmo.hit(point: HUDLayoutMath.Frame(controller.grip).origin, rayOrigin: aim.origin,
                                 rayDirection: -aim.axes.columns.2) {
                hovered[side] = h
            } else if grabs.allSatisfy({ $0 == nil }), let (piece, _) = pick(side: side, controller: controller, pieces: pieces) {
                selected = piece
            }
        }

        // The overlay: a ring on every piece either hand could pick up,
        // the held one's values beside it, the gizmo on the selected one.
        let now2 = LayoutStore.shared.current
        var marks: [Overlay.Mark] = []
        for (piece, placed) in pieces {
            let heldBy = grabs.firstIndex { $0?.piece == piece }
            let centre = HUDLayoutMath.Frame(placed.world).origin
            let shown = heldBy != nil || piece == selected
            let mirror = heldBy.flatMap { grabs[$0]?.mirror } ?? ({ if case .glove = piece { return mirrorGloves } else { return mirrorOrbs } }())
            marks.append(Overlay.Mark(piece: piece, centre: centre, radius: placed.radius, held: heldBy != nil,
                                      text: shown ? Self.values(of: piece, in: now2, mirror: mirror) : nil))
        }
        var gizmoMark: Overlay.GizmoMark?
        if let piece = selected, let placed = pieces[piece], let g = gizmoFor(piece, placed) {
            let active = grabs.compactMap { $0 }.first { $0.piece == piece && $0.handle != nil }?.handle
            gizmoMark = Overlay.GizmoMark(gizmo: g, hovered: Set(hovered.compactMap { $0 }), active: active)
        }
        overlayValue = Overlay(
            marks: marks,
            gizmo: gizmoMark,
            banner: Self.banner(nothingHere: pieces.isEmpty, mode: mode, worldSpace: worldSpace)
        )
    }

    /// The controls strip, exactly as the edit reads them.
    static func banner(nothingHere: Bool, mode: GizmoMode = .move, worldSpace: Bool = false) -> String {
        var text = "LAYOUT EDIT    gizmo: \(mode), \(worldSpace ? "room" : "local") axes\n"
            + "Point at or reach a piece to select it; its gizmo shows\n"
            + "Hold trigger on an arrow, ring or square: move along, turn about, or size only that\n"
            + "Hold the other hand's grip while dragging: snap (1 cm, 5 degrees, 5 per cent)\n"
            + "Cross or Square with nothing held: next gizmo mode (Move, Rotate, Scale)\n"
            + "One stick click tapped: local or room axes\n"
            + "Hold trigger on the piece itself: pick up and move freely (use the other hand for things on a hand)\n"
            + "Stick up or down while holding: size\n"
            + "Stick left or right while holding: tilt (right leans the top back)\n"
            + "Turn the controller while holding: the piece turns with it\n"
            + "Cross or Square while holding: mirror on or off\n"
            + "Circle or Triangle tapped while holding: put back as built\n"
            + "Hold both stick clicks, or tap Options: save and close\n"
            + "Hold Circle or Triangle with empty hands: close without saving"
        if nothingHere {
            text += "\nNothing to move while the game shows a menu: close the edit, leave the menu, open it again"
        }
        return text
    }

    private func setActive(_ on: Bool, save: Bool = true) {
        active = on
        rt_host_pad_pause(on ? 1 : 0)
        grabs = [nil, nil]
        menuSince = [nil, nil]
        upperSince = [nil, nil]
        if on {
            openedWith = LayoutStore.shared.current
            rt_host_log("layout edit: open")
        } else {
            overlayValue = Overlay()
            rt_host_log("layout edit: closed\(save ? "" : " without saving")")
            if save { LayoutStore.shared.save() }
            openedWith = nil
        }
    }

    private func release(_ side: Int) {
        if let grab = grabs[side] {
            rt_host_log("layout edit: put down the \(grab.piece): \(Self.values(of: grab.piece, in: LayoutStore.shared.current, mirror: grab.mirror))")
        }
        grabs[side] = nil
    }

    /// The piece a controller would pick up: the nearest within reach of
    /// its grip, or else the one its aim points closest to.
    private func pick(side: Int, controller: Controller, pieces: [HUDPiece: Placed]) -> (HUDPiece, Placed)? {
        let grip = HUDLayoutMath.Frame(controller.grip).origin
        let aim = HUDLayoutMath.Frame(controller.aim)
        let forward = -aim.axes.columns.2
        var best: (HUDPiece, Placed, Float)?
        for (piece, placed) in pieces {
            if piece.ridesHand, piece.hand == side { continue }
            if grabs.contains(where: { $0?.piece == piece }) { continue }
            let centre = HUDLayoutMath.Frame(placed.world).origin
            let reach = simd_length(centre - grip) - placed.radius
            var score = Float.greatestFiniteMagnitude
            if reach < 0.10 {
                score = reach
            } else {
                let ray = HUDLayoutMath.rayDistance(origin: aim.origin, direction: forward, to: centre)
                if ray.along > 0, ray.off < max(placed.radius * 1.5, ray.along * 0.05) {
                    score = 1 + ray.off / ray.along
                }
            }
            if score < (best?.2 ?? .greatestFiniteMagnitude) { best = (piece, placed, score) }
        }
        return best.map { ($0.0, $0.1) }
    }

    /// The gizmo for a piece as placed now: its own axes or the room's,
    /// in the edit's current mode. The reticle has none.
    private func gizmoFor(_ piece: HUDPiece, _ placed: Placed) -> Gizmo? {
        if case .reticle = piece { return nil }
        let f = HUDLayoutMath.Frame(placed.world)
        return Gizmo(centre: f.origin, axes: worldSpace ? matrix_identity_float3x3 : f.axes, radius: placed.radius, mode: mode)
    }

    /// A glove's grip for the weapon in hand, by the stick: a full stick
    /// held for a second moves either half the way.
    private func adjustGrip(_ h: Int, curl: Float, thumb: Float, dt: Float, mirror: Bool) {
        let dc = abs(curl) > 0.2 ? curl * 0.5 * dt : 0, dtb = abs(thumb) > 0.2 ? thumb * 0.5 * dt : 0
        guard dc != 0 || dtb != 0 else { return }
        let weapon = LayoutStore.shared.weapon
        LayoutStore.shared.update { l in
            var g = l.glove(h, weapon: weapon)
            g.curl = min(max(g.gripCurl + dc, 0), 1)
            g.thumb = min(max(g.thumbWrap + dtb, 0), 1)
            l.setGlove(h, weapon: weapon, g)
            if mirror {
                var o = l.glove(1 - h, weapon: weapon)
                o.curl = g.curl
                o.thumb = g.thumb
                l.setGlove(1 - h, weapon: weapon, o)
            }
        }
    }

    private func size(of piece: HUDPiece, in l: HUDLayout) -> Float {
        switch piece {
        case .orb(let h), .bodyOrb(let h): return l.orbs[h].size
        case .glove(let h): return l.glove(h, weapon: LayoutStore.shared.weapon).scale
        case .reticle: return l.reticleScale
        }
    }

    private func move(_ grab: Grab, controller: Controller, stickY: Float, dt: Float, pieces: [HUDPiece: Placed],
                      head: simd_float4x4, layout: HUDLayout, snap: Bool = false) {
        guard let placed = pieces[grab.piece] else { return }
        var now = HUDLayoutMath.held(grab.hold, controller: controller.grip)
        var grow = 1 + (abs(stickY) > 0.2 ? stickY : 0) * HUDLayout.stickRate * dt
        // A handle changes one thing only, from where the piece was taken.
        if let handle = grab.handle {
            let grip = HUDLayoutMath.Frame(controller.grip)
            let start = HUDLayoutMath.Frame(grab.startWorld)
            now = grab.startWorld
            grow = 1
            switch handle {
            case .axis(let i):
                let axis = grab.startAxes[i]
                let d = Gizmo.slide(from: grab.startGrip, to: grip.origin, axis: axis, snap: snap)
                now.columns.3 = SIMD4(start.origin + axis * d, 1)
            case .ring(let i):
                let axis = grab.startAxes[i]
                let angle = Gizmo.turn(centre: start.origin, axis: axis, startPoint: grab.startGrip, point: grip.origin,
                                       startTurn: grab.startTurn, nowTurn: simd_quatf(grip.axes), snap: snap)
                let r = simd_float3x3(simd_quatf(angle: angle, axis: axis)) * start.axes
                now = HUDLayoutMath.Frame(origin: start.origin, axes: r).matrix
            case .scale:
                let g = Gizmo(centre: start.origin, axes: grab.startAxes, radius: placed.radius, mode: .scale)
                let f = Gizmo.scaleFactor(from: grab.startGrip, to: grip.origin, outward: g.scalePoint - start.origin,
                                          radius: placed.radius, snap: snap)
                let current = size(of: grab.piece, in: layout)
                grow = current > 0 ? grab.startSize * f / current : 1
            }
        }
        let piece = grab.piece
        LayoutStore.shared.update { l in
            switch piece {
            case .orb(let h):
                var orb = HUDLayoutMath.orb(at: now, anchor: placed.anchor, facing: placed.facing,
                                            size: l.orbs[h].size * grow, mirrored: h == 0)
                orb.size = min(max(orb.size, HUDLayout.orbSizeRange.lowerBound), HUDLayout.orbSizeRange.upperBound)
                l.orbs[h] = orb
                if grab.mirror { l.orbs[1 - h] = orb }
            case .bodyOrb(let h):
                let at = HUDLayoutMath.bodyOrb(at: now, body: placed.anchor, mirrored: h == 0)
                l.bodyOrbs[h] = at
                if grab.mirror { l.bodyOrbs[1 - h] = at }
                let turn = HUDLayoutMath.orbDegrees(placed.facing.transpose * HUDLayoutMath.Frame(now).axes, mirrored: h == 0)
                var turns = l.bodyTurns ?? [l.orbs[0].turn, l.orbs[1].turn]
                turns[h] = turn
                if grab.mirror { turns[1 - h] = turn }
                l.bodyTurns = turns
                l.orbs[h].size *= grow
                if grab.mirror { l.orbs[1 - h].size = l.orbs[h].size }
            case .glove(let h):
                let fit = placed.anchor.matrix.inverse * now
                let weapon = LayoutStore.shared.weapon
                let was = l.glove(h, weapon: weapon)
                let glove = HUDLayoutMath.glove(fromFit: fit, scale: was.scale * grow, mirrored: h == 0).keepingGrip(of: was)
                l.setGlove(h, weapon: weapon, glove)
                if grab.mirror { l.setGlove(1 - h, weapon: weapon, glove) }
            case .reticle:
                // Nearer or further by ten times how far the controller has
                // moved along the view, so a short reach covers the range.
                let forward = -simd_normalize(SIMD3(head.columns.2.x, head.columns.2.y, head.columns.2.z))
                let moved = simd_dot(HUDLayoutMath.Frame(controller.grip).origin - grab.startGrip, forward)
                l.reticleDistance = grab.startDistance + moved * 10
                l.reticleScale *= grow
            }
        }
    }

    private func reset(_ piece: HUDPiece, mirror: Bool) {
        let d = LayoutStore.defaults
        LayoutStore.shared.update { l in
            switch piece {
            case .orb(let h):
                l.orbs[h] = d.orbs[h]
                if mirror { l.orbs[1 - h] = d.orbs[1 - h] }
            case .bodyOrb(let h):
                l.bodyOrbs[h] = d.bodyOrbs[h]
                if mirror { l.bodyOrbs[1 - h] = d.bodyOrbs[1 - h] }
                var turns = l.bodyTurns ?? [l.orbs[0].turn, l.orbs[1].turn]
                turns[h] = d.bodyTurn(h)
                if mirror { turns[1 - h] = d.bodyTurn(1 - h) }
                l.bodyTurns = turns
            case .glove(let h):
                // With a weapon in hand, back to the general place; with
                // none, the general place back to as built.
                let weapon = LayoutStore.shared.weapon
                let built = l.builtGloves ?? d.gloves
                l.resetGlove(h, weapon: weapon, to: built[h])
                if mirror { l.resetGlove(1 - h, weapon: weapon, to: built[1 - h]) }
            case .reticle:
                l.reticleDistance = d.reticleDistance
                l.reticleScale = d.reticleScale
            }
        }
        rt_host_log("layout edit: the \(piece) put back as built\(mirror && piece.partner != nil ? ", both sides" : "")")
    }

    /// A weapon's folder as a name to read: rifle_steyraug to "Steyraug
    /// (rifle)".
    static func weaponName(_ key: String) -> String {
        let parts = key.split(separator: "_").map(String.init)
        guard parts.count >= 2 else { return key.capitalized }
        return parts.dropFirst().joined(separator: " ").capitalized + " (\(parts[0]))"
    }

    /// A piece's values as the label shows them.
    static func values(of piece: HUDPiece, in l: HUDLayout, mirror: Bool) -> String {
        let m = piece.partner != nil ? (mirror ? "\nmirrored" : "\nthis side only") : ""
        func cm(_ v: SIMD3<Float>) -> String { String(format: "%.1f  %.1f  %.1f cm", v.x * 100, v.y * 100, v.z * 100) }
        func deg(_ v: SIMD3<Float>) -> String { String(format: "pitch %.0f  yaw %.0f  roll %.0f", v.x, v.y, v.z) }
        switch piece {
        case .orb(let h):
            let o = l.orbs[h]
            return "\(piece)\n\(cm(o.offset))\n\(deg(o.turn))\n" + String(format: "%.1f cm across", o.size * 100) + m
        case .bodyOrb(let h):
            return "\(piece)\n\(cm(l.bodyOrbs[h]))\n\(deg(l.bodyTurn(h)))\n" + String(format: "%.1f cm across", l.orbs[h].size * 100) + m
        case .glove(let h):
            let weapon = LayoutStore.shared.weapon
            let g = l.glove(h, weapon: weapon)
            let on = weapon.isEmpty ? "no weapon: every weapon without its own" : "for \(weaponName(weapon))"
            return "\(piece)\n\(on)\n\(cm(g.offset))\n\(deg(g.turn))\n"
                + String(format: "scale %.2f\ngrip %.0f%%, thumb %.0f%% (Scale mode: stick)", g.scale, g.gripCurl * 100, g.thumbWrap * 100) + m
        case .reticle:
            return "reticle\n" + String(format: "%.1f m out\nscale %.2f", l.reticleDistance, l.reticleScale)
        }
    }
}
