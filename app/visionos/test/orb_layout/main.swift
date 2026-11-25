// The orbs' porthole layout (Host/OrbLayout.swift), on the Mac:
//   swiftc -O app/visionos/Host/OrbLayout.swift app/visionos/test/orb_layout/main.swift -o orb_layout_test
// Every printed rectangle must lie inside the porthole's usable circle,
// whatever the name's length or the picture's shape.
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

func allInside(_ layout: OrbLayout.Layout) -> Bool {
    (layout.picture.map(OrbLayout.inside) ?? true) && layout.arcs.allSatisfy(OrbLayout.inside)
}

/// Lines on one arc follow each other left to right without touching;
/// arcs at different radii do not overlap radially.
func inOrderNoOverlap(_ arcs: [OrbLayout.Arc]) -> Bool {
    for (i, a) in arcs.enumerated() {
        for b in arcs[(i + 1)...] {
            let sameBand = abs(a.outer - b.outer) < 1e-4
            if sameBand {
                if b.start < a.end - 1e-5 { return false }
            } else if !(b.inner >= a.outer - 1e-5 || a.inner >= b.outer - 1e-5) {
                return false
            }
        }
    }
    return true
}

/// The picture's lower corners stay off the lettering's band: inside the
/// circle of its inner edge, or above where the band ends at the sides.
func clearOfText(_ layout: OrbLayout.Layout) -> Bool {
    guard let p = layout.picture, let inner = layout.arcs.map(\.inner).min() else { return true }
    let bottom = p.centre.y - p.half.y
    let corner = (p.half.x * p.half.x + bottom * bottom).squareRoot()
    let halfSpan = layout.arcs.map { max(-Float.pi / 2 - $0.start, $0.end + Float.pi / 2) }.max() ?? 0
    return corner <= inner + 1e-4 || bottom >= -inner * cos(halfSpan) - 1e-4
}

let name = OrbLayout.TextSize(preferred: 0.2, minimum: 0.11)
let ammo = OrbLayout.TextSize(preferred: 0.22, minimum: 0.14)
let icon = OrbLayout.Picture(uv: [0, 0, 1, 1], aspect: 1)

// A long one-line name, about 18 letters ("Mass Hallucination").
let long = OrbLayout.Line(uv: [0, 0, 1, 0.25], aspect: 11, gaps: [0.25])
let a = OrbLayout.layout(picture: icon, lines: [long], text: name)
check(allInside(a), "a long one-line name and the icon lie inside the porthole")
check(a.arcs.count == 1, "a long name stays one line, on the arc")
check(a.arcs.allSatisfy { $0.outer - $0.inner >= name.minimum - 1e-4 }, "its lettering is no smaller than legible")
check(clearOfText(a), "the icon sits clear of the name")
check(a.picture.map { $0.half.x >= 0.35 } == true, "the icon stays large (half \(a.picture?.half.x ?? 0))")
check(a.picture.map { $0.centre.y > 0 } == true, "the icon rises above the name")

// A name the game draws on two lines ("Bloodsuckers' Communion").
let two = [
    OrbLayout.Line(uv: [0, 0, 1, 0.1], aspect: 8),
    OrbLayout.Line(uv: [0, 0.1, 1, 0.2], aspect: 7.5),
]
let b = OrbLayout.layout(picture: icon, lines: two, text: name)
check(allInside(b), "a two-line name lies inside the porthole")
check(b.arcs.count == 2, "a two-line name keeps both its lines")
check(inOrderNoOverlap(b.arcs), "its lines are in order and do not overlap")
check(clearOfText(b), "the icon sits clear of both lines")
check(b.arcs.allSatisfy { $0.outer - $0.inner >= name.minimum - 1e-4 }, "both lines are no smaller than legible")

// Two short lines join onto one arc, in order.
let pair = [OrbLayout.Line(uv: [0, 0, 0.4, 0.1], aspect: 2), OrbLayout.Line(uv: [0, 0.1, 0.3, 0.2], aspect: 1.5)]
let c = OrbLayout.layout(picture: icon, lines: pair, text: name)
check(allInside(c), "two short lines lie inside the porthole")
check(c.arcs.count == 2 && abs(c.arcs[0].outer - c.arcs[1].outer) < 1e-4, "two short lines share one arc")
check(inOrderNoOverlap(c.arcs) && c.arcs[0].uv.y < c.arcs[1].uv.y, "the first line comes first along the arc")
check(c.arcs[0].outer - c.arcs[0].inner >= name.preferred - 1e-4, "short lines keep the size they want")

// A wide weapon (a rifle cropped to 4:1) and a short ammunition count.
let rifle = OrbLayout.Picture(uv: [0.05, 0.3, 0.95, 0.7], aspect: 4.2)
let count = OrbLayout.Line(uv: [0, 0, 0.3, 0.2], aspect: 2.2)
let d = OrbLayout.layout(picture: rifle, lines: [count], text: ammo)
check(allInside(d), "a wide weapon and its count lie inside the porthole")
check(d.picture.map { $0.half.x >= 0.6 } == true, "the wide weapon spans the porthole (half \(d.picture?.half.x ?? 0))")
check(clearOfText(d), "the weapon sits clear of its count")
check(d.arcs.first.map { $0.outer - $0.inner >= ammo.preferred - 1e-4 } == true, "the count keeps its large size")

// An impossibly long name: narrowed within the arc, never past it.
let huge = OrbLayout.Line(uv: [0, 0, 1, 0.1], aspect: 40)
let e = OrbLayout.layout(picture: icon, lines: [huge], text: name)
check(allInside(e), "an overlong name keeps within the arc and the circle")
let f = OrbLayout.layout(picture: icon, lines: [], text: name)
check(f.picture.map { abs($0.centre.y) < 1e-5 && $0.half.x > 0.45 } == true, "with no text the icon fills the porthole")

// The two names that clipped on the headset (2026-11-25), the game's
// letters about 0.6 of their height wide. Every glyph's corners, sampled
// along each arc at its foot and its top, inside the safe circle, the
// icon clear of the lettering, and the lettering no smaller than legible.
func glyphsInside(_ layout: OrbLayout.Layout) -> Bool {
    layout.arcs.allSatisfy { arc in
        (0...40).allSatisfy { k in
            let a = arc.start + (arc.end - arc.start) * Float(k) / 40
            return [arc.inner, arc.outer].allSatisfy { r in
                simd_length(SIMD2(r * cos(a), r * sin(a))) <= OrbLayout.usable + 1e-4 && r * sin(a) <= 1e-4
            }
        }
    }
}
func letters(_ count: Int) -> Float { Float(count) * 0.6 }
let names: [(String, [OrbLayout.Line])] = [
    ("Bloodsuckers' Communion", [
        OrbLayout.Line(uv: [0, 0, 1, 0.1], aspect: letters(13)), OrbLayout.Line(uv: [0, 0.1, 1, 0.2], aspect: letters(9)),
    ]),
    ("Mass Hallucination", [
        OrbLayout.Line(uv: [0, 0, 1, 0.1], aspect: letters(4)), OrbLayout.Line(uv: [0, 0.1, 1, 0.2], aspect: letters(13)),
    ]),
    ("Bloodsuckers' Communion on one line", [OrbLayout.Line(uv: [0, 0, 1, 0.1], aspect: letters(23))]),
    ("Mass Hallucination on one line", [OrbLayout.Line(uv: [0, 0, 1, 0.1], aspect: letters(18))]),
]
for (title, lines) in names {
    let g = OrbLayout.layout(picture: icon, lines: lines, text: name)
    check(allInside(g) && glyphsInside(g), "\(title): every glyph inside the safe circle")
    check(clearOfText(g), "\(title): the icon clear of the lettering")
    check(inOrderNoOverlap(g.arcs), "\(title): lines in order, no overlap")
    check(g.arcs.allSatisfy { $0.outer - $0.inner >= name.minimum - 1e-4 }, "\(title): no smaller than legible")
}

// The nameplate: the two names that crowded the porthole, one line and
// large when that fits, two when the game drew two and one is too long;
// always within the plate's widest, every line inside the plate.
for (title, lines) in names {
    guard let plate = OrbLayout.plate(lines: lines) else {
        check(false, "\(title): a nameplate is made")
        continue
    }
    let within = plate.lines.allSatisfy {
        abs($0.centre.x) + $0.half.x <= plate.half.x + 1e-4 && abs($0.centre.y) + $0.half.y <= plate.half.y + 1e-4
    }
    check(plate.half.x <= OrbLayout.plateWidest + 1e-4 && within, "\(title): the name lies on its plate, no wider than allowed")
    check(plate.lines.allSatisfy { $0.half.y * 2 >= 0.18 - 1e-4 }, "\(title): the plate's lettering is large (\(plate.lines.map { $0.half.y * 2 }))")
    check(plate.centreY + plate.half.y <= OrbLayout.plateTop + 1e-4, "\(title): the plate hangs below the base")
}
let shortPlate = OrbLayout.plate(lines: [OrbLayout.Line(uv: [0, 0, 1, 0.1], aspect: letters(6))])
check(shortPlate.map { $0.lines.count == 1 && abs($0.lines[0].half.y * 2 - OrbLayout.plateText) < 1e-4 } == true, "a short name is one line at full size")

// The read back: a 200 by 60 label with two words on one line (ink
// from 10 to 80 and 100 to 190, rows 20 to 39), and the same with a
// second line under it (rows 44 to 55, ink 30 to 150).
func label(_ spans: [(Int, Int, Int, Int)]) -> [UInt8] {
    var a = [UInt8](repeating: 0, count: 200 * 60)
    for (x0, x1, y0, y1) in spans {
        for y in y0...y1 { for x in x0...x1 where x % 6 != 5 { a[y * 200 + x] = 255 } }
    }
    return a
}
let one = OrbReads.lines(alpha: label([(10, 80, 20, 39), (100, 190, 20, 39)]), width: 200, height: 60, origin: .zero, size: [200, 60])
check(one.count == 1, "a one-line label reads as one line")
check(one.first.map { $0.gaps.contains { abs($0 - 0.49) < 0.06 } } == true, "its word gap is found near the middle (\(one.first?.gaps ?? []))")
let lettersOnly = one.first.map { $0.gaps.count == 1 } == true
check(lettersOnly, "the gaps between letters are not taken for word gaps")
let twoLines = OrbReads.lines(
    alpha: label([(10, 80, 20, 39), (100, 190, 20, 39), (30, 150, 44, 55)]), width: 200, height: 60, origin: .zero, size: [200, 60]
)
check(twoLines.count == 2, "a two-line label reads as two lines")
let alphaCrop = OrbReads.crop(alpha: label([(40, 160, 25, 35)]), width: 200, height: 60)
check(alphaCrop.map { abs($0.x - 0.195) < 0.01 && abs($0.z - 0.81) < 0.01 } == true, "a picture is cropped to its visible pixels")

print(failed == 0 ? "ALL PASS: \(passed) passed, 0 failed" : "\(failed) FAILED, \(passed) passed")
if failed != 0 { fatalError() }
