// The hand module offline, on the Mac: the rig and its poses on the CPU,
// the blender's easing, a poses file, then the renderer drawing the
// stand-in hand into an offscreen target, posed and hidden behind a scene's
// depth, with the pictures written out to look at. A modelled hand is
// loaded too when its path is given.
//
//   hands_test <out dir> [glove.usdz]

import Foundation
import ImageIO
import Metal
import simd

/// The glove grip defaults (HUDLayout.Glove, not compiled into this test).
enum HUDLayoutGrip {
    static let curl: Float = 0.85
    static let thumb: Float = 0.7
}
import UniformTypeIdentifiers

var passed = 0, failed = 0
func check(_ ok: Bool, _ what: String) {
    if ok {
        passed += 1
    } else {
        failed += 1
        print("FAIL: \(what)")
    }
}

let args = CommandLine.arguments
let outDir = URL(fileURLWithPath: args.count > 1 ? args[1] : FileManager.default.temporaryDirectory.path)
try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

// The rig.
let rig = ProceduralHand.rig()
let mesh = ProceduralHand.mesh()
check(rig.count == HandJoint.allCases.count, "the rig has every visionOS joint")
check(rig.parents.enumerated().allSatisfy { i, p in p.map { $0 < i } ?? true }, "parents come before children")
let rest = rig.restWorld
var restOK = true
for j in HandJoint.allCases where simd_distance(rest[j.rawValue].columns.3.xyz, ProceduralHand.restPositions[j]!) > 1e-5 {
    restOK = false
}
check(restOK, "rest world positions are the joints' places")
check(mesh.vertexCount > 1000 && mesh.indices.count % 3 == 0, "the stand-in mesh is made (\(mesh.vertexCount) vertices)")
check(mesh.weights.allSatisfy { abs($0.sum() - 1) < 1e-4 }, "every vertex's weights sum to one")

func skinned(_ shape: HandShape, rig: HandRig = rig, mesh: SkinnedMesh = mesh) -> [SIMD3<Float>] {
    mesh.skinned(palette: rig.palette(local: rig.local(for: shape.pose(on: rig))))
}
func tip(_ joint: HandJoint, _ shape: HandShape, rig: HandRig = rig) -> SIMD3<Float> {
    rig.world(local: rig.local(for: shape.pose(on: rig)))[rig.joint[joint]!].columns.3.xyz
}

let atRest = mesh.skinned(palette: rig.palette(local: rig.restLocal))
check(zip(atRest, mesh.positions).allSatisfy { simd_distance($0, $1) < 1e-5 }, "the rest pose leaves the mesh where it is")

let open = HandShape.open, fist = HandShape.fist
let indexOpen = tip(.indexFingerTip, open), indexFist = tip(.indexFingerTip, fist)
check(indexFist.x < indexOpen.x - 0.03, "a fist curls the index towards the palm (x \(indexOpen.x) to \(indexFist.x))")
check(indexFist.z > indexOpen.z + 0.05, "a fist brings the index tip back (z \(indexOpen.z) to \(indexFist.z))")
let pulled = HandShape.pistol.with(finger: 1, from: .trigger, by: 1)
check(simd_distance(tip(.middleFingerTip, pulled), tip(.middleFingerTip, .pistol)) < 1e-5, "the trigger moves the index alone")
let knuckle = ProceduralHand.restPositions[.indexFingerKnuckle]!
check(simd_distance(tip(.indexFingerTip, pulled), knuckle) < simd_distance(tip(.indexFingerTip, .pistol), knuckle) - 0.005,
      "the trigger pulls the index in")
let point = HandShape.point
check(tip(.indexFingerTip, point).z < tip(.middleFingerTip, point).z - 0.06, "pointing, the index stands out past the fist")
let thumbUp = tip(.thumbTip, .relaxed), thumbDown = tip(.thumbTip, .thumbDown)
check(thumbDown.x < thumbUp.x - 0.005, "the thumb goes over the palm to the controller's top (x \(thumbUp.x) to \(thumbDown.x))")
// Every vertex of a fist stays near the hand: nothing flies off.
let fistVerts = skinned(fist)
check(fistVerts.allSatisfy { simd_length($0) < 0.25 }, "a fist's vertices stay within the hand")

// The left hand: the right's mirror, posed the same, is the right's
// skinned mesh reflected.
let left = rig.mirrored()
let leftMesh = ProceduralHand.mirrored(mesh)
let l = skinned(.pistol, rig: left, mesh: leftMesh), r = skinned(.pistol)
check(zip(l, r).allSatisfy { simd_distance($0, SIMD3(-$1.x, $1.y, $1.z)) < 1e-4 }, "the left hand is the right's mirror, posed alike")

// The blender: a pressed trigger closes the finger over a few frames.
var blender = HandPoseBlender()
_ = blender.step(controls: HandControlState(), hold: .pistol, dt: 0.016)
let one = blender.step(controls: HandControlState(trigger: 1), hold: .pistol, dt: 0.016)
check(one.curl[1].x > HandShape.pistol.curl[1].x + 0.05 && one.curl[1].x < HandShape.trigger.curl[1].x - 0.05,
      "one frame in, the trigger's finger is part way (\(one.curl[1].x))")
var later = one
for _ in 0..<40 { later = blender.step(controls: HandControlState(trigger: 1), hold: .pistol, dt: 0.016) }
check(abs(later.curl[1].x - HandShape.trigger.curl[1].x) < 0.01, "after two thirds of a second it is all the way in")
for _ in 0..<5 { later = blender.step(controls: HandControlState(), hold: .point, dt: 0.016) }
check(later.curl[1].x > HandShape.point.curl[1].x + 0.05, "a change of hold eases, not snaps")
for _ in 0..<60 { later = blender.step(controls: HandControlState(), hold: .point, dt: 0.016) }
check(abs(later.curl[1].x - HandShape.point.curl[1].x) < 0.01, "and gets there")

// A poses file in the glove's form: per hand, per pose, each joint's
// local rotation.
let json = #"{"hands":{"right":{"poses":{"fist":{"indexFingerKnuckle":{"local":[0,0.7071068,0,0.7071068],"delta":[0,0,0,1]}}}},"left":{"poses":{"fist":{"indexFingerKnuckle":{"local":[0,-0.7071068,0,0.7071068]}}}}}}"#
if let file = HandPoseFile(json: Data(json.utf8), left: false), let pose = file.pose(named: "fist", on: rig),
   let leftFile = HandPoseFile(json: Data(json.utf8), left: true) {
    let i = rig.joint[.indexFingerKnuckle]!
    check(abs(simd_angle(pose.rotations[i]) - .pi / 2) < 1e-3, "a local pose replaces the joint's rotation")
    check(abs(simd_angle(pose.rotations[rig.joint[.middleFingerKnuckle]!])) < 1e-5, "joints it does not name stay at rest")
    check(leftFile.poses["fist"]?["indexFingerKnuckle"]?.imag.y ?? 0 < 0, "the left hand's poses are its own")
} else {
    check(false, "the poses file parses")
}

// The renderer, offscreen.
func perspective(fovY: Float, aspect: Float, near: Float) -> simd_float4x4 {
    // Reverse Z with the far plane at infinity: depth 1 at near, 0 far.
    let y = 1 / tan(fovY / 2), x = y / aspect
    return simd_float4x4(SIMD4(x, 0, 0, 0), SIMD4(0, y, 0, 0), SIMD4(0, 0, 0, -1), SIMD4(0, 0, near, 0))
}
func lookAt(eye: SIMD3<Float>, at: SIMD3<Float>, up: SIMD3<Float>) -> simd_float4x4 {
    let f = simd_normalize(at - eye), s = simd_normalize(simd_cross(f, up)), u = simd_cross(s, f)
    let r = simd_float4x4(SIMD4(s.x, u.x, -f.x, 0), SIMD4(s.y, u.y, -f.y, 0), SIMD4(s.z, u.z, -f.z, 0), SIMD4(0, 0, 0, 1))
    var t = matrix_identity_float4x4
    t.columns.3 = SIMD4(-eye, 1)
    return r * t
}

guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
    print("no Metal device: \(passed) passed, rendering skipped")
    exit(failed == 0 ? 0 : 1)
}
let size = 320
let colourDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm_srgb, width: size, height: size, mipmapped: false)
colourDesc.usage = [.renderTarget, .shaderRead]
colourDesc.storageMode = .shared
let depthDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float, width: size, height: size, mipmapped: false)
depthDesc.usage = .renderTarget
depthDesc.storageMode = .private
let colour = device.makeTexture(descriptor: colourDesc)!, depth = device.makeTexture(descriptor: depthDesc)!
let renderer = try HandRenderer(device: device, colourFormat: .bgra8Unorm_srgb, depthFormat: .depth32Float)
let rightMesh = HandMesh(device: device, mesh: mesh, jointCount: rig.count)!
let leftGPU = HandMesh(device: device, mesh: leftMesh, jointCount: left.count)!

// The camera behind and above the back of the hand, looking at the palm's
// middle from the thumb's side.
let eye = SIMD3<Float>(0.22, 0.12, 0.08)
let view = lookAt(eye: eye, at: SIMD3(0, 0, -0.05), up: SIMD3(0, 1, 0))
let viewProjection = perspective(fovY: 0.9, aspect: 1, near: 0.01) * view

func render(_ draws: [(HandMesh, HandRig, HandShape, simd_float4x4)], scene: SceneDepth? = nil) -> (Int, [UInt8]) {
    let pass = MTLRenderPassDescriptor()
    pass.colorAttachments[0].texture = colour
    pass.colorAttachments[0].loadAction = .clear
    pass.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
    pass.colorAttachments[0].storeAction = .store
    pass.depthAttachment.texture = depth
    pass.depthAttachment.loadAction = .clear
    pass.depthAttachment.clearDepth = 0
    pass.depthAttachment.storeAction = .dontCare
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeRenderCommandEncoder(descriptor: pass)!
    for (gpu, r, shape, model) in draws {
        let palette = r.palette(local: r.local(for: shape.pose(on: r)))
        renderer.draw(gpu, palette: palette, model: model, viewProjection: viewProjection, eye: eye, scene: scene, encoder: enc)
    }
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    var bytes = [UInt8](repeating: 0, count: size * size * 4)
    colour.getBytes(&bytes, bytesPerRow: size * 4, from: MTLRegionMake2D(0, 0, size, size), mipmapLevel: 0)
    var covered = 0
    for i in stride(from: 3, to: bytes.count, by: 4) where bytes[i] > 0 { covered += 1 }
    return (covered, bytes)
}

func writePNG(_ bytes: [UInt8], _ name: String) {
    var rgba = bytes
    for i in stride(from: 0, to: rgba.count, by: 4) { rgba.swapAt(i, i + 2) }
    let provider = CGDataProvider(data: Data(rgba) as CFData)!
    let image = CGImage(
        width: size, height: size, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: size * 4,
        space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
        provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent
    )!
    let url = outDir.appendingPathComponent(name)
    if let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) {
        CGImageDestinationAddImage(dest, image, nil)
        CGImageDestinationFinalize(dest)
    }
}

// Cameras for the gloves, in the module's axes (fingers -Z, palm -X for
// the right): from the player's eye, above and behind the hand looking
// along it, and from the side of the back of the hand.
func gloveEye(left: Bool) -> (simd_float4x4, SIMD3<Float>) {
    let s: Float = left ? -1 : 1
    let e = SIMD3<Float>(0.05 * s, 0.2, 0.28)
    let v = lookAt(eye: e, at: SIMD3(0, 0, -0.08), up: SIMD3(0, 1, 0))
    return (perspective(fovY: 0.75, aspect: 1, near: 0.01) * v, e)
}
func gloveSide(left: Bool) -> (simd_float4x4, SIMD3<Float>) {
    let s: Float = left ? -1 : 1
    let e = SIMD3<Float>(0.3 * s, 0.08, -0.04)
    let v = lookAt(eye: e, at: SIMD3(0, 0, -0.07), up: SIMD3(0, 1, 0))
    return (perspective(fovY: 0.75, aspect: 1, near: 0.01) * v, e)
}
func renderPalette(_ gpu: HandMesh, palette: [simd_float4x4], camera: (simd_float4x4, SIMD3<Float>)) -> (Int, [UInt8]) {
    let pass = MTLRenderPassDescriptor()
    pass.colorAttachments[0].texture = colour
    pass.colorAttachments[0].loadAction = .clear
    pass.colorAttachments[0].clearColor = MTLClearColor(red: 0.18, green: 0.2, blue: 0.24, alpha: 1)
    pass.colorAttachments[0].storeAction = .store
    pass.depthAttachment.texture = depth
    pass.depthAttachment.loadAction = .clear
    pass.depthAttachment.clearDepth = 0
    pass.depthAttachment.storeAction = .dontCare
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeRenderCommandEncoder(descriptor: pass)!
    renderer.draw(gpu, palette: palette, model: matrix_identity_float4x4, viewProjection: camera.0, eye: camera.1, encoder: enc)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    var bytes = [UInt8](repeating: 0, count: size * size * 4)
    colour.getBytes(&bytes, bytesPerRow: size * 4, from: MTLRegionMake2D(0, 0, size, size), mipmapLevel: 0)
    var covered = 0
    let bg = (Int(bytes[0]), Int(bytes[1]), Int(bytes[2]))
    for i in stride(from: 0, to: bytes.count, by: 4)
    where abs(Int(bytes[i]) - bg.0) > 3 || abs(Int(bytes[i + 1]) - bg.1) > 3 || abs(Int(bytes[i + 2]) - bg.2) > 3 {
        covered += 1
    }
    return (covered, bytes)
}

let identity = matrix_identity_float4x4
let (openCover, openPixels) = render([(rightMesh, rig, .open, identity)])
let (fistCover, fistPixels) = render([(rightMesh, rig, .fist, identity)])
let (pistolCover, pistolPixels) = render([(rightMesh, rig, HandShape.pistol.with(finger: 1, from: .trigger, by: 1), identity)])
writePNG(openPixels, "hand_open.png")
writePNG(fistPixels, "hand_fist.png")
writePNG(pistolPixels, "hand_trigger.png")
check(openCover > 3000, "the open hand draws (\(openCover) pixels)")
check(fistCover > 1500 && fistCover < openCover - 500, "a fist draws, smaller than the open hand (\(fistCover))")
check(pistolCover > 1500, "the pulled trigger draws (\(pistolCover))")
var lit = 0
for i in stride(from: 0, to: openPixels.count, by: 4) where openPixels[i + 3] > 0 {
    if Int(openPixels[i]) + Int(openPixels[i + 1]) + Int(openPixels[i + 2]) > 30 { lit += 1 }
}
check(lit > openCover / 3, "the hand is lit, not black (\(lit) of \(openCover))")

// Both hands side by side, the left mirrored by its own rig and mesh.
var shiftRight = identity, shiftLeft = identity
shiftRight.columns.3 = SIMD4(0, 0, 0.06, 1)
shiftLeft.columns.3 = SIMD4(0, 0, -0.16, 1)
let (bothCover, bothPixels) = render([(rightMesh, rig, .relaxed, shiftRight), (leftGPU, left, .relaxed, shiftLeft)])
writePNG(bothPixels, "hands_both.png")
check(bothCover > openCover, "both hands draw (\(bothCover))")

// Behind a scene: a wall 5 cm in front of the hand over the image's left
// half hides that half; one far behind hides nothing; one nearer than the
// ignore distance is taken for what is held and hides nothing.
func sceneDepth(_ metres: Float, leftHalfOnly: Bool) -> MTLTexture {
    let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r16Float, width: 64, height: 64, mipmapped: false)
    d.textureType = .type2DArray
    d.arrayLength = 2
    let t = device.makeTexture(descriptor: d)!
    var texels = [Float16](repeating: 0, count: 64 * 64)
    for y in 0..<64 { for x in 0..<64 where !leftHalfOnly || x < 32 { texels[y * 64 + x] = Float16(metres) } }
    for s in 0..<2 {
        t.replace(region: MTLRegionMake2D(0, 0, 64, 64), mipmapLevel: 0, slice: s, withBytes: texels, bytesPerRow: 128,
                  bytesPerImage: 128 * 64)
    }
    return t
}
// The scene image is this test's camera: the same view, tangents of its
// field.
let tanHalf = tan(Float(0.9) / 2)
let handDistance = simd_distance(eye, SIMD3(0, 0, -0.05))
func scene(_ texture: MTLTexture, ignore: Float = 0) -> SceneDepth {
    SceneDepth(texture: texture, slice: 0, imageFromOrigin: view, tangents: SIMD4(repeating: tanHalf), bias: 0.01, ignoreNearer: ignore)
}
let wall = sceneDepth(0.1, leftHalfOnly: true)
let (hiddenCover, hiddenPixels) = render([(rightMesh, rig, .open, identity)], scene: scene(wall))
writePNG(hiddenPixels, "hand_behind_wall.png")
check(hiddenCover < openCover * 3 / 4 && hiddenCover > openCover / 8, "a nearer wall over half the view hides part (\(hiddenCover) of \(openCover))")
var leftHalfLit = 0
for y in 0..<size { for x in 0..<(size / 2) where hiddenPixels[(y * size + x) * 4 + 3] > 0 { leftHalfLit += 1 } }
check(leftHalfLit == 0, "nothing of the hand shows over the wall (\(leftHalfLit) pixels)")
let (farCover, _) = render([(rightMesh, rig, .open, identity)], scene: scene(sceneDepth(handDistance + 1, leftHalfOnly: false)))
check(farCover == openCover, "a scene behind the hand hides nothing")
let (heldCover, _) = render([(rightMesh, rig, .open, identity)], scene: scene(wall, ignore: handDistance))
check(heldCover == openCover, "a scene nearer than the ignore distance hides nothing")

// The modelled gloves, when their folder is given: each pose of the
// model's own, both hands, from the player's eye and from a three quarter
// view, written out to look at.
if args.count > 2 {
    let dir = URL(fileURLWithPath: args[2])
    for (left, name) in [(false, "glove_right.usdz"), (true, "glove_left.usdz")] {
        let url = dir.appendingPathComponent(name)
        guard FileManager.default.fileExists(atPath: url.path) else { continue }
        do {
            let model = try HandModelLoader.load(url: url, left: left)
            let matched = HandJoint.allCases.filter { model.rig.joint[$0] != nil }.count
            print("model \(model.source): \(model.mesh.vertexCount) vertices, \(model.mesh.indices.count / 3) triangles, "
                  + "\(model.rig.count) joints, \(matched) of \(HandJoint.allCases.count) visionOS joints matched, colour "
                  + "\(model.baseColourTexture != nil), normals \(model.normalTexture != nil), roughness "
                  + "\(model.roughnessTexture != nil), poses \(model.poses?.poses.keys.sorted() ?? [])")
            check(model.mesh.vertexCount > 0 && matched >= 25, "\(name) loads with visionOS's joints")
            check(model.poses?.poses.count ?? 0 >= 9, "\(name) has its poses")
            check(model.normalTexture != nil && model.baseColourTexture != nil, "\(name) has its colour and normal maps")
            let signs = Set(model.mesh.tangents.map { $0.w > 0 })
            print("  tangent signs: \(signs.sorted { !$0 && $1 })")
            guard let gpu = HandMesh(device: device, model: model) else { check(false, "\(name) uploads"); continue }
            let restPalette = model.rig.palette(local: model.rig.restLocal)
            let restSkinned = model.mesh.skinned(palette: restPalette)
            check(zip(restSkinned, model.mesh.positions).allSatisfy { simd_distance($0, $1) < 1e-4 },
                  "\(name) at rest skins to its own vertices")
            // In the module's axes the right palm faces -X: its vertices'
            // normals on the palm side point that way.
            let fitted = restSkinned.map { (model.fit * SIMD4($0, 1)).xyz }
            let bounds = fitted.reduce((SIMD3<Float>(repeating: .infinity), SIMD3<Float>(repeating: -.infinity))) {
                (simd_min($0.0, $1), simd_max($0.1, $1))
            }
            print("  fitted bounds \(bounds.0) to \(bounds.1)")
            check(bounds.0.z < -0.12 && bounds.1.z < 0.12, "\(name)'s fingers run along -Z from the wrist")
            let thumbTip = (model.fit * model.rig.restWorld[model.rig.joint[.thumbTip]!].columns.3).xyz
            check(thumbTip.y > 0.01, "\(name)'s thumb is up in the module's axes (\(thumbTip))")
            var blender = HandPoseBlender()
            for pose in ["relaxed", "trigger", "grip", "fist", "point", "pistol_grip", "rifle_grip", "melee_grip"] {
                guard let p = model.poses?.pose(named: pose, on: model.rig) else { check(false, "\(name) has \(pose)"); continue }
                let palette = model.rig.palette(local: model.rig.local(for: p))
                for (tag, camera) in [("eye", gloveEye(left: left)), ("side", gloveSide(left: left))] {
                    let (c, px) = renderPalette(gpu, palette: palette, camera: camera)
                    writePNG(px, "glove_\(left ? "left" : "right")_\(pose)_\(tag).png")
                    check(c > 2000, "\(name) draws \(pose) from the \(tag) (\(c) pixels)")
                }
            }
            // The blender on the file: a pulled trigger on the pistol grip.
            _ = blender.step(controls: HandControlState(), hold: .pistol, dt: 0.016, file: model.poses!, rig: model.rig)
            var pulled: HandJointPose?
            for _ in 0..<40 {
                pulled = blender.step(controls: HandControlState(trigger: 1), hold: .pistol, dt: 0.016, file: model.poses!, rig: model.rig)
            }
            if let pulled, let trig = model.poses?.pose(named: "trigger", on: model.rig) {
                let i = model.rig.joint[.indexFingerKnuckle]!, m = model.rig.joint[.middleFingerKnuckle]!
                check(abs(simd_angle(pulled.rotations[i] * trig.rotations[i].inverse)) < 0.02, "\(name): the trigger pulls the index to the file's")
                // On a gun the lower three close round the grip by the glove's
                // curl (0.85 by default) toward the fist, whatever the trigger.
                let grip = model.poses!.pose(named: "pistol_grip", on: model.rig)!
                let fist = model.poses!.pose(named: "fist", on: model.rig)!
                let fingers = model.rig.names.map { HandJoint(name: $0)?.finger }
                let closed = grip.blended(to: fist, weights: fingers.map { f in
                    guard let f else { return 0 }
                    return f >= 2 ? HUDLayoutGrip.curl : f == 0 ? HUDLayoutGrip.thumb : 0
                })
                check(abs(simd_angle(pulled.rotations[m] * closed.rotations[m].inverse)) < 0.02,
                      "\(name): the middle stays closed round the grip by the glove's curl")
                check(abs(simd_angle(closed.rotations[m] * grip.rotations[m].inverse)) > 0.05,
                      "\(name): the middle closes further than the file's pistol grip")
                let (c, px) = renderPalette(gpu, palette: model.rig.palette(local: model.rig.local(for: pulled)), camera: gloveEye(left: left))
                writePNG(px, "glove_\(left ? "left" : "right")_pistol_pulled_eye.png")
                check(c > 2000, "\(name) draws with the trigger pulled")
            }
        } catch {
            check(false, "\(name) loads: \(error)")
        }
    }
}

print("hands: \(passed) passed, \(failed) failed; pictures in \(outDir.path)")
exit(failed == 0 ? 0 : 1)
