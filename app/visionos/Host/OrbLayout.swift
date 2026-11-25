import Foundation
import simd

/// What goes on an orb's porthole, and where. Pure arithmetic, so it is
/// tested on the Mac (`app/visionos/test/orb_layout/main.swift`).
///
/// The porthole is a circle in its own plane, the glass's radius 1, and
/// everything printed on it must lie inside `OrbLayout.usable` (the
/// bezel ring's inner edge less a margin). The picture fills that circle,
/// centred. The text is set along the circle's lower inside edge, like
/// lettering engraved round a bezel, upright to the arc and reading left
/// to right, over a soft dark band so it reads over the picture: a long
/// name fits by running further round, not by wrapping (wrapped names
/// looked wrong on the headset, 2026-11-24), and the text shrinks only
/// when it would run past `maxArc`. A name the game draws on two lines
/// is joined onto one arc when that fits, else its second line takes the
/// outer arc and its first the arc just inside it.
enum OrbLayout {
    /// The bezel's inner edge is at about 0.785 of the glass's radius
    /// (a 52 degree porthole); 0.74 leaves a margin inside it.
    /// The bezel stands proud of the glass, so seen at an angle it hides
    /// a band inside its own edge: 0.74 let the ends and foot of a lower
    /// line slip under it on the headset (2026-11-25), 0.66 does not.
    static let usable: Float = 0.66
    /// The most of the circle a line may take, centred at the bottom:
    /// 140 degrees, or up to 180 when one line would otherwise have to
    /// shrink below legible.
    static let maxArc: Float = 140 * .pi / 180
    static let widestArc: Float = .pi
    /// The space between two lines joined onto one arc, in line heights,
    /// and between two arcs.
    static let joinSpace: Float = 0.45
    static let arcGap: Float = 0.02

    /// A line of the game's text on the label texture: its rectangle
    /// there (u0, v0, u1, v1), its width over its height in pixels, and
    /// where its word gaps are, as fractions of its width.
    struct Line: Equatable {
        var uv: SIMD4<Float>
        var aspect: Float
        var gaps: [Float] = []
    }

    struct Picture: Equatable {
        var uv: SIMD4<Float>
        var aspect: Float
    }

    /// The text's height on the porthole, in the glass's radii: what it
    /// wants, and the least it may be.
    struct TextSize {
        var preferred: Float
        var minimum: Float
    }

    /// The picture's rectangle: centre and half size (x across, y up).
    struct Rect: Equatable {
        var centre: SIMD2<Float>
        var half: SIMD2<Float>
        var uv: SIMD4<Float>
    }

    /// A line set on an arc: its lettering lies between radius `inner`
    /// (the letters' tops, towards the middle) and `outer` (their foot),
    /// from angle `start` to `end` (radians, anticlockwise from +x; the
    /// bottom is -pi/2, so start is the left end), its source on its
    /// texture across that.
    struct Arc: Equatable {
        var inner: Float
        var outer: Float
        var start: Float
        var end: Float
        var uv: SIMD4<Float>
    }

    struct Layout: Equatable {
        var picture: Rect?
        var arcs: [Arc]
    }

    static func layout(picture: Picture?, lines: [Line], text: TextSize) -> Layout {
        var out = layoutText(lines: lines, text: text)
        // The picture fills the circle when there is no text; with text
        // it rises clear of the lettering's band (a headset run,
        // 2026-11-24: a long name crowded the icon).
        if let picture {
            let inner = out.arcs.map(\.inner).min()
            let halfSpan = out.arcs.map { max(-Float.pi / 2 - $0.start, $0.end + Float.pi / 2) }.max() ?? 0
            out.picture = inner.map { above(picture, textInner: $0, halfSpan: halfSpan) } ?? fill(picture)
        }
        return out
    }

    /// The gap kept between the picture and the lettering's inner edge.
    static let pictureGap: Float = 0.04

    /// The picture as large as it can be inside the circle with its
    /// lower corners clear of the text: either within the circle the
    /// lettering's inner edge makes, or above where the band ends at the
    /// sides. It may rise off centre to grow.
    static func above(_ picture: Picture, textInner: Float, halfSpan: Float) -> Rect {
        let a = max(picture.aspect, 1e-3)
        let clear = textInner - pictureGap
        let bandTop = -clear * cos(halfSpan)
        func fits(_ h: Float, _ c: Float) -> Bool {
            let w = a * h
            guard w * w + (c + h) * (c + h) <= usable * usable else { return false }
            let bottom = c - h
            return w * w + bottom * bottom <= clear * clear || bottom >= bandTop
        }
        var best = Rect(centre: .zero, half: SIMD2(a * 0.05, 0.05), uv: picture.uv)
        var c: Float = 0
        while c <= usable {
            var lo: Float = 0, hi = usable
            for _ in 0..<24 {
                let mid = (lo + hi) / 2
                if fits(mid, c) { lo = mid } else { hi = mid }
            }
            if lo > best.half.y { best = Rect(centre: SIMD2(0, c), half: SIMD2(a * lo, lo), uv: picture.uv) }
            c += 0.005
        }
        return best
    }

    /// The text: short lines joined onto one arc at the size they want;
    /// else two lines on two arcs when both fit at a legible size within
    /// `maxArc`; else every line joined onto one arc, as large as
    /// that arc allows; a line that would go below legible within 140
    /// degrees runs further round, up to 180, before it shrinks more.
    /// Every arc's foot is at or inside `usable`, so nothing is clipped.
    private static func layoutText(lines: [Line], text: TextSize) -> Layout {
        var out = Layout(picture: nil, arcs: [])
        guard !lines.isEmpty else { return out }
        let joined = lines.reduce(0) { $0 + $1.aspect } + joinSpace * Float(lines.count - 1)
        // Short lines join onto one arc, at the size they want.
        if lines.count >= 2, height(aspect: joined, outer: usable, arc: maxArc) >= text.preferred - 1e-5 {
            out.arcs = set(lines, outer: usable, height: text.preferred, arc: maxArc)
            return out
        }
        if lines.count >= 2 {
            let last = lines[lines.count - 1], first = lines[lines.count - 2]
            var h = min(text.preferred, height(aspect: last.aspect, outer: usable, arc: maxArc))
            for _ in 0..<12 {
                h = min(h, height(aspect: first.aspect, outer: usable - h - arcGap, arc: maxArc))
            }
            if h >= text.minimum - 1e-5 {
                out.arcs = set([first], outer: usable - h - arcGap, height: h, arc: maxArc)
                    + set([last], outer: usable, height: h, arc: maxArc)
                return out
            }
        }
        var arc = maxArc
        var h = height(aspect: joined, outer: usable, arc: arc)
        if h < text.minimum {
            // Further round, up to the widest, to stay legible.
            arc = min(widestArc, text.minimum * joined / (usable - text.minimum / 2))
            h = height(aspect: joined, outer: usable, arc: arc)
        }
        out.arcs = set(lines, outer: usable, height: min(text.preferred, h), arc: arc)
        return out
    }

    /// The picture as large as the circle holds, centred, its shape kept.
    static func fill(_ picture: Picture) -> Rect {
        let a = max(picture.aspect, 1e-3)
        let halfHigh = usable / (a * a + 1).squareRoot()
        return Rect(centre: .zero, half: SIMD2(halfHigh * a, halfHigh), uv: picture.uv)
    }

    /// The tallest a line of this shape can be on an arc whose foot is at
    /// `outer` and still take no more than `maxArc`: its length along the
    /// arc's middle is aspect times height.
    static func height(aspect: Float, outer: Float, arc: Float = maxArc) -> Float {
        arc * outer / (max(aspect, 1e-3) + arc / 2)
    }

    /// Lines one after another on one arc, centred at the bottom, at a
    /// height that keeps them within `arc` (the caller's), letters set at
    /// their own shape.
    private static func set(_ lines: [Line], outer: Float, height h: Float, arc: Float) -> [Arc] {
        let total = lines.reduce(0) { $0 + $1.aspect } + joinSpace * Float(lines.count - 1)
        let middle = outer - h / 2
        let span = min(total * h / middle, arc)
        let perAspect = span / total
        var angle = -Float.pi / 2 - span / 2
        var arcs: [Arc] = []
        for (k, line) in lines.enumerated() {
            if k > 0 { angle += joinSpace * perAspect }
            let next = angle + line.aspect * perAspect
            arcs.append(Arc(inner: outer - h, outer: outer, start: angle, end: next, uv: line.uv))
            angle = next
        }
        return arcs
    }

    /// The brass nameplate under the blood orb's base: the power's name
    /// engraved and lit from behind, one line and large, on a plate sized
    /// to it up to `plateWidest` (1.4 times the orb's width, in the
    /// glass's radii, as a half), below the base at `plateTop`. A name too
    /// long for that at `plateText` goes on two lines if the game drew it
    /// on two, then shrinks as it must. A long name on the arc in the
    /// porthole read as a thin illegible band (headset run, 2026-11-25).
    struct Plate: Equatable {
        /// The plate's centre height and half size, in the orb's frame.
        var centreY: Float
        var half: SIMD2<Float>
        /// The lettering, centred on the plate (x across, y up from its
        /// middle), each line's rectangle and its source.
        var lines: [Rect]
    }

    static let plateWidest: Float = 1.4
    static let plateTop: Float = -1.26
    static let plateText: Float = 0.3
    static let plateMargin = SIMD2<Float>(0.14, 0.08)
    static let plateLineGap: Float = 0.05

    static func plate(lines: [Line]) -> Plate? {
        guard !lines.isEmpty else { return nil }
        let room = plateWidest - plateMargin.x
        let joined = lines.reduce(0) { $0 + $1.aspect } + joinSpace * Float(lines.count - 1)
        // One line when it fits large, or when the game drew one.
        let stacked = lines.count >= 2 && plateText * joined / 2 > room
        let rows: [[Line]] = stacked ? lines.map { [$0] } : [lines]
        let widths = rows.map { row in row.reduce(0) { $0 + $1.aspect } + joinSpace * Float(row.count - 1) }
        let widest = widths.max() ?? 1
        let h = min(plateText, room * 2 / widest)
        let half = SIMD2(min(widest * h / 2 + plateMargin.x, plateWidest),
                         (h * Float(rows.count) + plateLineGap * Float(rows.count - 1)) / 2 + plateMargin.y)
        var out: [Rect] = []
        var top = half.y - plateMargin.y
        for (row, width) in zip(rows, widths) {
            var x = -width * h / 2
            for (k, line) in row.enumerated() {
                if k > 0 { x += joinSpace * h }
                let w = line.aspect * h
                out.append(Rect(centre: SIMD2(x + w / 2, top - h / 2), half: SIMD2(w / 2, h / 2), uv: line.uv))
                x += w
            }
            top -= h + plateLineGap
        }
        return Plate(centreY: plateTop - half.y, half: half, lines: out)
    }

    /// Whether a rectangle lies inside the usable circle: its corners do.
    static func inside(_ rect: Rect) -> Bool {
        let far = SIMD2(abs(rect.centre.x) + rect.half.x, max(abs(rect.centre.y + rect.half.y), abs(rect.centre.y - rect.half.y)))
        return simd_length(far) <= usable + 1e-4
    }

    /// Whether an arc's lettering lies inside the usable circle and within
    /// the lower arc allowed.
    static func inside(_ arc: Arc) -> Bool {
        let half = widestArc / 2 + 1e-4
        return arc.outer <= usable + 1e-4 && arc.inner > 0 && arc.start >= -Float.pi / 2 - half
            && arc.end <= -Float.pi / 2 + half && arc.end > arc.start
    }
}

/// What the CPU reads back from the orbs' prints (`ScreenCompositor.readAlpha`):
/// each picture's crop to what it shows, keyed by the game's texture, and
/// each label's lines and word gaps, keyed by where the label sits on its
/// texture. Written from command buffer completions, read by the
/// compositor's thread, so behind a lock.
final class OrbReads: @unchecked Sendable {
    private let lock = NSLock()
    private var crops: [(source: ObjectIdentifier, crop: SIMD4<Float>)?] = [nil, nil]
    private var read: [(uv: SIMD4<Float>, lines: [OrbLayout.Line])?] = [nil, nil]
    private var asked: [(uv: SIMD4<Float>, frame: UInt64)?] = [nil, nil]
    private var pending = [false, false]

    func crop(side: Int, of source: AnyObject?) -> SIMD4<Float>? {
        guard let source else { return nil }
        lock.lock()
        defer { lock.unlock() }
        guard let made = crops[side], made.source == ObjectIdentifier(source) else { return nil }
        return made.crop
    }

    func setCrop(side: Int, source: ObjectIdentifier, crop: SIMD4<Float>?) {
        lock.lock()
        crops[side] = crop.map { (source, $0) }
        lock.unlock()
    }

    func lines(side: Int, uv: SIMD4<Float>) -> [OrbLayout.Line]? {
        lock.lock()
        defer { lock.unlock() }
        guard let made = read[side], made.uv == uv, !made.lines.isEmpty else { return nil }
        return made.lines
    }

    /// Whether a side's label should be read now: its place changed, or
    /// 20 frames have passed, and no read of it is in flight.
    func wantsRead(side: Int, uv: SIMD4<Float>, frame: UInt64) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if pending[side] { return false }
        if let last = asked[side], last.uv == uv, frame < last.frame + 20 { return false }
        asked[side] = (uv, frame)
        pending[side] = true
        return true
    }

    func setLines(side: Int, uv: SIMD4<Float>, lines: [OrbLayout.Line]) {
        lock.lock()
        read[side] = (uv, lines)
        pending[side] = false
        lock.unlock()
    }

    /// The rectangle (u0, v0, u1, v1) holding a picture's visible pixels,
    /// a texel of margin round them; nil for a picture with none.
    static func crop(alpha: [UInt8], width: Int, height: Int) -> SIMD4<Float>? {
        var x0 = width, y0 = height, x1 = -1, y1 = -1
        for y in 0..<height {
            for x in 0..<width where alpha[y * width + x] > 25 {
                x0 = min(x0, x)
                x1 = max(x1, x)
                y0 = min(y0, y)
                y1 = max(y1, y)
            }
        }
        guard x1 >= x0, y1 >= y0 else { return nil }
        let w = Float(width), h = Float(height)
        return SIMD4(
            Float(max(x0 - 1, 0)) / w, Float(max(y0 - 1, 0)) / h, Float(min(x1 + 2, width)) / w,
            Float(min(y1 + 2, height)) / h
        )
    }

    /// A label's lines: runs of rows with ink (a gap of two rows or less
    /// joins them), each cropped to its ink with a little margin, and its
    /// word gaps, runs of clear columns wider than about a quarter of its
    /// height. Rectangles in the label texture's uv (size), the read
    /// having started at origin.
    static func lines(alpha: [UInt8], width: Int, height: Int, origin: SIMD2<Int>, size: SIMD2<Int>) -> [OrbLayout.Line] {
        let ink: UInt8 = 77
        var rows = [Bool](repeating: false, count: height)
        for y in 0..<height {
            for x in 0..<width where alpha[y * width + x] > ink {
                rows[y] = true
                break
            }
        }
        var runs: [(Int, Int)] = []
        var y = 0
        while y < height {
            guard rows[y] else {
                y += 1
                continue
            }
            var end = y
            while end + 1 < height && (rows[end + 1] || (end + 3 < height && (rows[end + 2] || rows[end + 3]))) {
                end += 1
            }
            if end - y + 1 >= 3 { runs.append((y, end)) }
            y = end + 1
        }
        var lines: [OrbLayout.Line] = []
        for (top, bottom) in runs {
            var cols = [Bool](repeating: false, count: width)
            for row in top...bottom {
                for x in 0..<width where alpha[row * width + x] > ink { cols[x] = true }
            }
            guard let first = cols.firstIndex(of: true), let last = cols.lastIndex(of: true) else { continue }
            let lineHeight = bottom - top + 1
            let span = Float(last - first + 1)
            var gaps: [Float] = []
            var x = first
            while x <= last {
                if cols[x] {
                    x += 1
                    continue
                }
                var end = x
                while end + 1 <= last && !cols[end + 1] { end += 1 }
                if end - x + 1 >= max(3, Int(Float(lineHeight) * 0.28)) {
                    gaps.append((Float(x + end) / 2 - Float(first) + 0.5) / span)
                }
                x = end + 1
            }
            let px0 = origin.x + max(first - 2, 0), px1 = origin.x + min(last + 3, width)
            let py0 = origin.y + max(top - 2, 0), py1 = origin.y + min(bottom + 3, height)
            let uv = SIMD4<Float>(
                Float(px0) / Float(size.x), Float(py0) / Float(size.y), Float(px1) / Float(size.x), Float(py1) / Float(size.y)
            )
            // The gaps as fractions of the cropped width, margins included.
            let cropped = Float(px1 - px0), lead = Float(origin.x + first - px0)
            let fractions = gaps.map { ($0 * span + lead) / cropped }
            lines.append(OrbLayout.Line(uv: uv, aspect: Float(px1 - px0) / Float(py1 - py0), gaps: fractions))
        }
        return lines
    }
}
