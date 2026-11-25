// A soft target lock: of the things a player could act on, which one the
// aim means. Nearest valid candidate inside a cone around the aim, held
// against flicker by hysteresis, cycled left and right by a flick. Pure
// value types with no engine, renderer or tracking dependency, so any
// port can feed it candidates and read back one stable id; what a
// candidate is and what selecting it does belong to that port's adapter.
import simd

/// One thing that could be selected, in whatever world space the rays use.
struct SoftTargetCandidate: Sendable, Equatable {
    /// Stable across frames for the same thing (an entity number, say).
    var id: Int
    var centre: SIMD3<Float>
    /// Widens the thing's angular size, so a big target near the edge of
    /// the cone still counts.
    var radius: Float
    /// False for things that exist but cannot be picked now (dead, allied).
    var selectable: Bool
}

struct SoftTargetRay: Sendable, Equatable {
    var origin: SIMD3<Float>
    /// Need not be normalised.
    var direction: SIMD3<Float>
}

struct SoftTargetConfig: Sendable, Equatable {
    /// A candidate enters when its angle from the aim is inside this.
    var coneDegrees: Float = 10
    /// The held one is kept until it is outside this wider cone, so it does
    /// not drop out and back in at the edge.
    var exitConeDegrees: Float = 13
    /// A challenger must be this much closer to the aim than the held one...
    var challengeMarginDegrees: Float = 2
    /// ...for this long, before it takes over.
    var challengeSeconds: Double = 0.15
    /// Angles within this are a tie, broken by distance and then id.
    var tieDegrees: Float = 0.05
    /// Candidates further than this are ignored.
    var maxDistance: Float = .greatestFiniteMagnitude
    /// Cycling considers candidates inside this cone, and a cycled pick is
    /// kept until it leaves it.
    var cycleConeDegrees: Float = 30
    /// Which way is up, for "left" and "right" when cycling.
    var up: SIMD3<Float> = [0, 1, 0]
    /// Seconds for the highlight to come fully up after a pick.
    var riseSeconds: Double = 0.12
    /// Seconds for it to fade after the selection is lost.
    var fallSeconds: Double = 0.2
}

enum SoftTargetCycle: Sendable {
    case left, right
}

/// The selector's whole state. Feed it a frame with `update`; read
/// `selected` and `strength`.
struct SoftTargetSelector: Sendable {
    var config: SoftTargetConfig

    /// The id the aim means, or nil.
    private(set) var selected: Int?
    /// 0 to 1, for drawing the highlight. Resets to 0 when the pick changes.
    private(set) var strength: Float = 0
    /// The last id that was selected, so a fade can be drawn on it.
    private(set) var fading: Int?

    private var challenger: Int?
    private var challengerSince: Double = 0
    private var pinned = false
    private var lastTime: Double?
    private var lastCandidates: [SoftTargetCandidate] = []
    private var lastAim: SoftTargetRay?
    private var lastGaze: SoftTargetRay?

    init(config: SoftTargetConfig = SoftTargetConfig()) {
        self.config = config
    }

    /// Angle in radians between a ray and a candidate, less the candidate's
    /// own angular radius, never below zero. Nil when behind the ray's
    /// origin or out of range.
    static func angle(_ ray: SoftTargetRay, to c: SoftTargetCandidate, maxDistance: Float) -> (angle: Float, distance: Float)? {
        let to = c.centre - ray.origin
        let distance = simd_length(to)
        guard distance > 1e-6, distance <= maxDistance else { return nil }
        let dir = simd_normalize(ray.direction)
        let cosine = simd_clamp(simd_dot(dir, to / distance), -1, 1)
        guard cosine > 0 else { return nil }
        let widen = c.radius > 0 ? asin(min(c.radius / distance, 1)) : 0
        return (max(acos(cosine) - widen, 0), distance)
    }

    /// The angle used for picking: the aim's, or the gaze's when a gaze is
    /// given and it is nearer.
    private func score(_ c: SoftTargetCandidate, aim: SoftTargetRay, gaze: SoftTargetRay?) -> (angle: Float, distance: Float)? {
        let byAim = Self.angle(aim, to: c, maxDistance: config.maxDistance)
        guard let gaze, let byGaze = Self.angle(gaze, to: c, maxDistance: config.maxDistance) else { return byAim }
        guard let byAim else { return byGaze }
        return byGaze.angle < byAim.angle ? byGaze : byAim
    }

    /// Orders candidates by angle, then distance, then id, so two at the
    /// same angle always resolve the same way.
    private func better(_ a: (c: SoftTargetCandidate, angle: Float, distance: Float),
                        _ b: (c: SoftTargetCandidate, angle: Float, distance: Float)) -> Bool {
        let tie = config.tieDegrees * .pi / 180
        if abs(a.angle - b.angle) > tie { return a.angle < b.angle }
        if a.distance != b.distance { return a.distance < b.distance }
        return a.c.id < b.c.id
    }

    /// One frame: the time in seconds, the aim, an optional gaze and every
    /// candidate this frame. Returns the selected id.
    @discardableResult
    mutating func update(time: Double, aim: SoftTargetRay, gaze: SoftTargetRay? = nil,
                         candidates: [SoftTargetCandidate]) -> Int? {
        let dt = lastTime.map { max(time - $0, 0) } ?? 0
        lastTime = time
        lastCandidates = candidates
        lastAim = aim
        lastGaze = gaze

        let enter = config.coneDegrees * .pi / 180
        let exit = max(config.exitConeDegrees, config.coneDegrees) * .pi / 180
        let cycleCone = config.cycleConeDegrees * .pi / 180

        var scored: [(c: SoftTargetCandidate, angle: Float, distance: Float)] = []
        for c in candidates where c.selectable {
            if let s = score(c, aim: aim, gaze: gaze) { scored.append((c, s.angle, s.distance)) }
        }
        let inside = scored.filter { $0.angle <= enter }.sorted(by: better)
        let best = inside.first

        let previous = selected
        if let held = selected, let h = scored.first(where: { $0.c.id == held }),
           h.angle <= (pinned ? max(cycleCone, exit) : exit) {
            // Still valid. A challenger takes over only after beating it by
            // the margin for the whole challenge time; a cycled pick is not
            // challenged until it leaves the cycle cone.
            if !pinned, let b = best, b.c.id != held,
               b.angle + config.challengeMarginDegrees * .pi / 180 < h.angle {
                if challenger != b.c.id {
                    challenger = b.c.id
                    challengerSince = time
                }
                if time - challengerSince >= config.challengeSeconds {
                    selected = b.c.id
                    challenger = nil
                }
            } else {
                challenger = nil
            }
        } else {
            // Gone, dead or out of the cone: the next best takes over at once.
            selected = best?.c.id
            challenger = nil
            pinned = false
        }

        if selected != previous {
            fading = previous
            // A new pick starts its highlight from nothing; a loss keeps the
            // level so it can fade.
            if selected != nil { strength = 0 }
        }
        if selected != nil {
            strength = min(strength + Float(dt / max(config.riseSeconds, 1e-3)), 1)
        } else {
            strength = max(strength - Float(dt / max(config.fallSeconds, 1e-3)), 0)
        }
        return selected
    }

    /// A flick: move the selection to the next candidate to the left or
    /// right of the current one, in screen order around the aim, among
    /// those inside the cycle cone. Stays put at either end. The pick is
    /// then held until it leaves the cycle cone. Returns the new id.
    @discardableResult
    mutating func cycle(_ way: SoftTargetCycle) -> Int? {
        guard let aim = lastAim else { return selected }
        let forward = simd_normalize(aim.direction)
        var right = simd_cross(forward, config.up)
        if simd_length(right) < 1e-5 { right = simd_cross(forward, [0, 0, 1]) }
        right = simd_normalize(right)
        let cycleCone = config.cycleConeDegrees * .pi / 180

        // Horizontal bearing of each candidate around the aim.
        var bearings: [(id: Int, bearing: Float)] = []
        for c in lastCandidates where c.selectable {
            guard let s = score(c, aim: aim, gaze: lastGaze), s.angle <= cycleCone else { continue }
            let to = c.centre - aim.origin
            bearings.append((c.id, atan2(simd_dot(to, right), simd_dot(to, forward))))
        }
        guard !bearings.isEmpty else { return selected }
        bearings.sort { $0.bearing != $1.bearing ? $0.bearing < $1.bearing : $0.id < $1.id }

        let next: Int?
        if let held = selected, let i = bearings.firstIndex(where: { $0.id == held }) {
            let j = way == .right ? i + 1 : i - 1
            next = bearings.indices.contains(j) ? bearings[j].id : held
        } else {
            // Nothing held: the nearest one on that side of the aim.
            next = way == .right ? (bearings.first { $0.bearing >= 0 } ?? bearings.last)?.id
                                 : (bearings.last { $0.bearing <= 0 } ?? bearings.first)?.id
        }
        if next != selected {
            fading = selected
            selected = next
            strength = 0
        }
        pinned = selected != nil
        challenger = nil
        return selected
    }

    /// Drops the selection, for a port that wants the lock released.
    mutating func clear() {
        if selected != nil { fading = selected }
        selected = nil
        challenger = nil
        pinned = false
    }
}
