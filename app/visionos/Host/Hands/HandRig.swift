import Foundation
import simd

// A hand's skeleton, its poses and a stand-in mesh, with nothing in it that
// knows which app draws it: no Metal, no input, no game. The joints are
// visionOS's own (`HandSkeleton.JointName`), so a hand model rigged with
// those names, a controller and a tracked hand all speak one language.
//
// Conventions, right hand, model space in metres: the wrist at the origin,
// the fingers along -Z, the thumb up along +Y and the palm facing -X, as a
// right hand held out to shake another's. A left hand is the mirror image
// in X. A finger curls towards the palm: about +Y for the right hand.

/// visionOS's hand joints, in `HandSkeleton.JointName`'s order and spelling.
enum HandJoint: Int, CaseIterable, Sendable {
    case wrist
    case thumbKnuckle, thumbIntermediateBase, thumbIntermediateTip, thumbTip
    case indexFingerMetacarpal, indexFingerKnuckle, indexFingerIntermediateBase, indexFingerIntermediateTip, indexFingerTip
    case middleFingerMetacarpal, middleFingerKnuckle, middleFingerIntermediateBase, middleFingerIntermediateTip, middleFingerTip
    case ringFingerMetacarpal, ringFingerKnuckle, ringFingerIntermediateBase, ringFingerIntermediateTip, ringFingerTip
    case littleFingerMetacarpal, littleFingerKnuckle, littleFingerIntermediateBase, littleFingerIntermediateTip, littleFingerTip
    case forearmWrist, forearmArm

    var name: String { String(describing: self) }

    init?(name: String) {
        guard let joint = Self.allCases.first(where: { $0.name == name }) else { return nil }
        self = joint
    }

    var parent: HandJoint? {
        switch self {
        case .wrist: return nil
        case .thumbKnuckle, .indexFingerMetacarpal, .middleFingerMetacarpal, .ringFingerMetacarpal,
             .littleFingerMetacarpal, .forearmWrist:
            return .wrist
        case .forearmArm: return .forearmWrist
        default: return HandJoint(rawValue: rawValue - 1)
        }
    }

    /// The finger a joint belongs to: 0 the thumb, 1 index to 4 little, nil
    /// for the wrist and forearm.
    var finger: Int? {
        switch rawValue {
        case 1...4: return 0
        case 5...9: return 1
        case 10...14: return 2
        case 15...19: return 3
        case 20...24: return 4
        default: return nil
        }
    }

    /// The finger's joints that bend, base first: the knuckle and the two
    /// between the knuckle and the tip (the thumb's three likewise).
    static func bending(finger: Int) -> [HandJoint] {
        switch finger {
        case 0: return [.thumbKnuckle, .thumbIntermediateBase, .thumbIntermediateTip]
        case 1: return [.indexFingerKnuckle, .indexFingerIntermediateBase, .indexFingerIntermediateTip]
        case 2: return [.middleFingerKnuckle, .middleFingerIntermediateBase, .middleFingerIntermediateTip]
        case 3: return [.ringFingerKnuckle, .ringFingerIntermediateBase, .ringFingerIntermediateTip]
        default: return [.littleFingerKnuckle, .littleFingerIntermediateBase, .littleFingerIntermediateTip]
        }
    }
}

/// A skeleton as a mesh was bound to it: its joints in the mesh's own order,
/// each with its parent, its rest transform under the parent, and the
/// inverse of where it was in model space when the skin was bound. `joint`
/// maps visionOS's joints to it; a model may lack some (no forearm) and have
/// others (a strap) that the poses leave at rest.
struct HandRig: Sendable {
    var names: [String]
    var parents: [Int?]
    var restLocal: [simd_float4x4]
    var inverseBind: [simd_float4x4]
    var joint: [HandJoint: Int]
    /// The axis a finger curls about and the thumb's, in model space at
    /// rest, so a pose made of curls (`HandShape`) bends any rig the right
    /// way. Mirrored with the hand.
    var curlAxis: SIMD3<Float>
    var thumbAxis: SIMD3<Float>
    var spreadAxis: SIMD3<Float>
    /// The thumb's swing across the palm, down over the fingers.
    var opposeAxis: SIMD3<Float> = SIMD3(0, 0, 1)

    var count: Int { names.count }

    /// Model space transforms for local ones, parents first in the rig's
    /// order (a rig from a file is checked into that order when loaded).
    func world(local: [simd_float4x4]) -> [simd_float4x4] {
        var out = local
        for i in 0..<count {
            if let p = parents[i] { out[i] = out[p] * local[i] }
        }
        return out
    }

    /// The skinning palette: each joint's posed model transform times its
    /// inverse bind.
    func palette(local: [simd_float4x4]) -> [simd_float4x4] {
        let w = world(local: local)
        return (0..<count).map { w[$0] * inverseBind[$0] }
    }

    var restWorld: [simd_float4x4] { world(local: restLocal) }

    /// Local transforms for a pose: each joint's rest transform with its
    /// rotation replaced by the pose's, its translation kept.
    func local(for pose: HandJointPose) -> [simd_float4x4] {
        (0..<count).map { i in
            var m = simd_float4x4(pose.rotations[i])
            m.columns.3 = restLocal[i].columns.3
            return m
        }
    }

    /// Orders joints so every parent comes before its children, remapping
    /// parents; returns the old index of each new one. A rig loaded from a
    /// file may list them in any order.
    mutating func sortParentsFirst() -> [Int] {
        var order: [Int] = []
        var placed = [Bool](repeating: false, count: count)
        while order.count < count {
            var progressed = false
            for i in 0..<count where !placed[i] {
                if let p = parents[i], !placed[p] { continue }
                order.append(i)
                placed[i] = true
                progressed = true
            }
            if !progressed {
                // A cycle: break it by rooting what is left.
                for i in 0..<count where !placed[i] {
                    parents[i] = nil
                    order.append(i)
                    placed[i] = true
                }
            }
        }
        var newIndex = [Int](repeating: 0, count: count)
        for (n, o) in order.enumerated() { newIndex[o] = n }
        names = order.map { names[$0] }
        parents = order.map { parents[$0].map { newIndex[$0] } }
        restLocal = order.map { restLocal[$0] }
        inverseBind = order.map { inverseBind[$0] }
        joint = joint.mapValues { newIndex[$0] }
        return order
    }

    /// The same rig for the other hand: every transform reflected in X, the
    /// axes with it (an axis of rotation turns the other way in a mirror).
    func mirrored() -> HandRig {
        let m = simd_float4x4(diagonal: SIMD4(-1, 1, 1, 1))
        var out = self
        out.restLocal = restLocal.map { m * $0 * m }
        out.inverseBind = inverseBind.map { m * $0 * m }
        let flip = { (a: SIMD3<Float>) in -SIMD3(-a.x, a.y, a.z) }
        out.curlAxis = flip(curlAxis)
        out.thumbAxis = flip(thumbAxis)
        out.spreadAxis = flip(spreadAxis)
        out.opposeAxis = flip(opposeAxis)
        return out
    }
}

/// A pose: each rig joint's rotation under its parent.
struct HandJointPose: Sendable {
    var rotations: [simd_quatf]

    init(rotations: [simd_quatf]) { self.rotations = rotations }

    init(rest rig: HandRig) {
        rotations = rig.restLocal.map { simd_quatf(simd_float3x3(columns: ($0.columns.0.xyz, $0.columns.1.xyz, $0.columns.2.xyz))) }
    }

    /// Per joint spherical blend towards `other` by `weight[joint]`.
    func blended(to other: HandJointPose, weights: [Float]) -> HandJointPose {
        var out = self
        for i in rotations.indices where weights[i] > 0 {
            out.rotations[i] = simd_slerp(rotations[i], other.rotations[i], min(max(weights[i], 0), 1))
        }
        return out
    }

    func blended(to other: HandJointPose, by weight: Float) -> HandJointPose {
        blended(to: other, weights: [Float](repeating: weight, count: rotations.count))
    }
}

/// A pose described by how far each finger curls rather than by rotations:
/// any rig takes it (`pose(on:)`), and two blend by blending numbers.
/// Curls are radians at the finger's three bending joints, base first.
struct HandShape: Sendable, Equatable {
    var curl: [SIMD3<Float>]  // thumb, index, middle, ring, little
    var spread: [Float]       // each finger's fan about the knuckle, radians
    var thumbAcross: Float    // the thumb swung over the palm, radians

    static let open = HandShape(curl: Array(repeating: .zero, count: 5), spread: [0, 0.08, 0, -0.06, -0.14], thumbAcross: 0)
    static let relaxed = HandShape(
        curl: [SIMD3(0.15, 0.2, 0.15), SIMD3(0.3, 0.35, 0.2), SIMD3(0.38, 0.42, 0.24), SIMD3(0.45, 0.48, 0.26),
               SIMD3(0.52, 0.5, 0.28)],
        spread: [0, 0.05, 0, -0.04, -0.09], thumbAcross: 0.15
    )
    /// Wrapped round a handle: the controller's, or a weapon's grip.
    static let fist = HandShape(
        curl: [SIMD3(0.35, 0.55, 0.5), SIMD3(1.35, 1.55, 0.95), SIMD3(1.4, 1.6, 0.95), SIMD3(1.42, 1.6, 0.95),
               SIMD3(1.45, 1.55, 0.9)],
        spread: [0, 0, 0, 0, -0.03], thumbAcross: 0.55
    )
    static let point = HandShape(
        curl: [SIMD3(0.35, 0.55, 0.45), SIMD3(0.05, 0.05, 0.03), SIMD3(1.4, 1.6, 0.95), SIMD3(1.42, 1.6, 0.95),
               SIMD3(1.45, 1.55, 0.9)],
        spread: [0, 0.04, 0, 0, -0.03], thumbAcross: 0.5
    )
    /// A pistol: three fingers round the grip, the index along the trigger
    /// guard ready, the thumb down the far side.
    static let pistol = HandShape(
        curl: [SIMD3(0.3, 0.35, 0.2), SIMD3(0.55, 0.85, 0.4), SIMD3(1.35, 1.55, 0.95), SIMD3(1.4, 1.58, 0.95),
               SIMD3(1.42, 1.55, 0.9)],
        spread: [0, 0.02, 0, 0, -0.03], thumbAcross: 0.35
    )
    /// A rifle's pistol grip, the same with the wrist's fingers a touch
    /// looser; the off hand holds the fore end (`support`).
    static let rifle = HandShape(
        curl: [SIMD3(0.3, 0.35, 0.2), SIMD3(0.55, 0.85, 0.4), SIMD3(1.3, 1.5, 0.9), SIMD3(1.35, 1.52, 0.9),
               SIMD3(1.38, 1.5, 0.85)],
        spread: [0, 0.02, 0, 0, -0.03], thumbAcross: 0.3
    )
    static let support = HandShape(
        curl: [SIMD3(0.2, 0.25, 0.15), SIMD3(0.9, 1.0, 0.6), SIMD3(0.95, 1.05, 0.6), SIMD3(1.0, 1.05, 0.6),
               SIMD3(1.05, 1.0, 0.55)],
        spread: [0, 0.03, 0, -0.02, -0.05], thumbAcross: 0.15
    )
    /// A hilt: everything round it, the thumb over the fingers.
    static let melee = HandShape(
        curl: [SIMD3(0.45, 0.7, 0.6), SIMD3(1.3, 1.5, 0.9), SIMD3(1.4, 1.6, 0.95), SIMD3(1.42, 1.6, 0.95),
               SIMD3(1.45, 1.55, 0.9)],
        spread: [0, 0, 0, 0, -0.02], thumbAcross: 0.7
    )
    /// The trigger pulled: only the index's curl matters.
    static let trigger = HandShape(
        curl: [SIMD3(0.3, 0.35, 0.2), SIMD3(1.05, 1.3, 0.7), SIMD3(1.35, 1.55, 0.95), SIMD3(1.4, 1.58, 0.95),
               SIMD3(1.42, 1.55, 0.9)],
        spread: [0, 0, 0, 0, -0.03], thumbAcross: 0.35
    )
    /// The thumb laid on a stick or a face button on top of the controller.
    static let thumbDown = HandShape(
        curl: [SIMD3(0.45, 0.6, 0.45), SIMD3(0.3, 0.35, 0.2), SIMD3(0.38, 0.42, 0.24), SIMD3(0.45, 0.48, 0.26),
               SIMD3(0.52, 0.5, 0.28)],
        spread: [0, 0.05, 0, -0.04, -0.09], thumbAcross: 0.6
    )

    static let named: [String: HandShape] = [
        "open": .open, "relaxed": .relaxed, "fist": .fist, "point": .point, "pistol": .pistol, "rifle": .rifle,
        "support": .support, "melee": .melee, "trigger": .trigger, "thumbDown": .thumbDown,
    ]

    /// The rotations on a rig: each bending joint's rest rotation turned by
    /// its curl about the rig's axis as it lies at that joint at rest, the
    /// knuckles fanned by the spread. Joints the shape does not name keep
    /// their rest rotation.
    func pose(on rig: HandRig) -> HandJointPose {
        var pose = HandJointPose(rest: rig)
        let rest = rig.restWorld
        func turn(_ joint: HandJoint, about axis: SIMD3<Float>, by angle: Float) {
            guard let i = rig.joint[joint], angle != 0 else { return }
            // The axis in the joint's own rest frame, so the turn happens
            // after its rest rotation, about the joint itself.
            let r = simd_float3x3(columns: (rest[i].columns.0.xyz, rest[i].columns.1.xyz, rest[i].columns.2.xyz))
            let local = simd_normalize(r.inverse * axis)
            pose.rotations[i] = pose.rotations[i] * simd_quatf(angle: angle, axis: local)
        }
        for finger in 0..<5 {
            let joints = HandJoint.bending(finger: finger)
            let axis = finger == 0 ? rig.thumbAxis : rig.curlAxis
            if finger == 0 {
                turn(joints[0], about: rig.opposeAxis, by: thumbAcross)
            } else {
                turn(joints[0], about: rig.spreadAxis, by: spread[finger])
            }
            for k in 0..<3 { turn(joints[k], about: axis, by: curl[finger][k]) }
        }
        return pose
    }

    static func mix(_ a: HandShape, _ b: HandShape, _ t: Float) -> HandShape {
        HandShape(
            curl: zip(a.curl, b.curl).map { simd_mix($0, $1, SIMD3(repeating: t)) },
            spread: zip(a.spread, b.spread).map { $0 + ($1 - $0) * t },
            thumbAcross: a.thumbAcross + (b.thumbAcross - a.thumbAcross) * t
        )
    }

    /// One finger of b over a, by t; the rest of a as it is.
    func with(finger: Int, from b: HandShape, by t: Float) -> HandShape {
        var out = self
        out.curl[finger] = simd_mix(curl[finger], b.curl[finger], SIMD3(repeating: t))
        out.spread[finger] = spread[finger] + (b.spread[finger] - spread[finger]) * t
        if finger == 0 { out.thumbAcross = thumbAcross + (b.thumbAcross - thumbAcross) * t }
        return out
    }
}

/// Poses a model ships with (`poses.json` beside it), by name: for each
/// hand, each joint's rotation under its parent ("local", the rest
/// transform's rotation replaced) as x, y, z, w. The file names its hands
/// "right" and "left" under "hands"; joints are matched by name, and a
/// joint the file does not name keeps its rest rotation. A name the file
/// has is used in place of the curl shapes (`HandShape`).
struct HandPoseFile: Sendable {
    var poses: [String: [String: simd_quatf]]

    init(poses: [String: [String: simd_quatf]]) { self.poses = poses }

    init?(json data: Data, left: Bool) {
        guard let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let hands = root["hands"] as? [String: Any],
              let hand = hands[left ? "left" : "right"] as? [String: Any],
              let body = hand["poses"] as? [String: Any]
        else { return nil }
        var poses: [String: [String: simd_quatf]] = [:]
        for (name, value) in body {
            guard let joints = value as? [String: Any] else { continue }
            var rotations: [String: simd_quatf] = [:]
            for (joint, forms) in joints {
                guard let v = (forms as? [String: Any])?["local"] as? [Double], v.count == 4 else { continue }
                rotations[joint] = simd_normalize(simd_quatf(ix: Float(v[0]), iy: Float(v[1]), iz: Float(v[2]), r: Float(v[3])))
            }
            if !rotations.isEmpty { poses[name] = rotations }
        }
        guard !poses.isEmpty else { return nil }
        self.poses = poses
    }

    func pose(named name: String, on rig: HandRig) -> HandJointPose? {
        guard let rotations = poses[name] else { return nil }
        var pose = HandJointPose(rest: rig)
        for (i, jointName) in rig.names.enumerated() {
            if let q = rotations[jointName] { pose.rotations[i] = q }
        }
        return pose
    }
}

/// Vertices ready for the GPU: model space position and normal, uv, four
/// joint indices into the rig and their weights.
struct SkinnedMesh: Sendable {
    var positions: [SIMD3<Float>] = []
    var normals: [SIMD3<Float>] = []
    var uvs: [SIMD2<Float>] = []
    /// The texture's right along the surface, w the sign that makes the
    /// bitangent cross(normal, tangent) * w (a mirrored UV layout has -1).
    var tangents: [SIMD4<Float>] = []
    var joints: [SIMD4<UInt16>] = []
    var weights: [SIMD4<Float>] = []
    var indices: [UInt32] = []

    var vertexCount: Int { positions.count }

    /// Skinned on the CPU, for tests: each vertex through its joints'
    /// palette by weight.
    func skinned(palette: [simd_float4x4]) -> [SIMD3<Float>] {
        positions.indices.map { v in
            var p = SIMD4<Float>.zero
            let j = joints[v], w = weights[v]
            for k in 0..<4 where w[k] > 0 {
                p += w[k] * (palette[Int(j[k])] * SIMD4(positions[v], 1))
            }
            return p.xyz
        }
    }

    /// Interleaved as the renderer reads it (`HandRendererLayout.stride`):
    /// position and normal packed, uv, joints as four shorts, weights and
    /// tangent packed. A mesh without tangents gets any perpendicular one.
    func interleaved() -> [UInt8] {
        var out = [UInt8](repeating: 0, count: vertexCount * HandRendererLayout.stride)
        out.withUnsafeMutableBytes { raw in
            for v in 0..<vertexCount {
                let base = raw.baseAddress! + v * HandRendererLayout.stride
                let f = base.assumingMemoryBound(to: Float.self)
                f[0] = positions[v].x; f[1] = positions[v].y; f[2] = positions[v].z
                f[3] = normals[v].x; f[4] = normals[v].y; f[5] = normals[v].z
                f[6] = uvs[v].x; f[7] = uvs[v].y
                let s = (base + 32).assumingMemoryBound(to: UInt16.self)
                s[0] = joints[v].x; s[1] = joints[v].y; s[2] = joints[v].z; s[3] = joints[v].w
                let w = (base + 40).assumingMemoryBound(to: Float.self)
                w[0] = weights[v].x; w[1] = weights[v].y; w[2] = weights[v].z; w[3] = weights[v].w
                let t = v < tangents.count ? tangents[v] : SkinnedMesh.anyTangent(normals[v])
                w[4] = t.x; w[5] = t.y; w[6] = t.z; w[7] = t.w
            }
        }
        return out
    }
}

/// The interleaved vertex's layout, shared by the mesh and the shader.
enum HandRendererLayout {
    static let stride = 72
}

extension SkinnedMesh {
    static func anyTangent(_ n: SIMD3<Float>) -> SIMD4<Float> {
        let helper: SIMD3<Float> = abs(n.y) < 0.9 ? SIMD3(0, 1, 0) : SIMD3(1, 0, 0)
        let c = simd_cross(helper, n)
        return simd_length(c) > 1e-6 ? SIMD4(simd_normalize(c), 1) : SIMD4(1, 0, 0, 1)
    }
}

/// A stand-in hand made in code: a capsule for each bone and a rounded
/// palm, on a rig with visionOS's joints, so the whole path can be built,
/// tested and seen before a modelled hand exists. A right hand; mirror it
/// for the left (`HandRig.mirrored`, with the model's X flipped).
enum ProceduralHand {
    /// Where each joint sits at rest, model space, metres.
    static let restPositions: [HandJoint: SIMD3<Float>] = {
        var p: [HandJoint: SIMD3<Float>] = [.wrist: .zero, .forearmWrist: .zero, .forearmArm: SIMD3(0, 0, 0.22)]
        // Thumb: from the base of the palm out and forward, up off the palm.
        p[.thumbKnuckle] = SIMD3(-0.018, 0.022, -0.028)
        let thumbDir = simd_normalize(SIMD3<Float>(-0.32, 0.42, -0.85))
        p[.thumbIntermediateBase] = p[.thumbKnuckle]! + thumbDir * 0.042
        p[.thumbIntermediateTip] = p[.thumbIntermediateBase]! + thumbDir * 0.032
        p[.thumbTip] = p[.thumbIntermediateTip]! + thumbDir * 0.027
        // Fingers stacked up the palm's edge, index on top.
        let rows: [(HandJoint, Float, Float, SIMD3<Float>)] = [
            (.indexFingerMetacarpal, 0.022, -0.094, SIMD3(0.046, 0.027, 0.022)),
            (.middleFingerMetacarpal, 0.004, -0.098, SIMD3(0.05, 0.031, 0.024)),
            (.ringFingerMetacarpal, -0.014, -0.092, SIMD3(0.047, 0.029, 0.023)),
            (.littleFingerMetacarpal, -0.031, -0.082, SIMD3(0.037, 0.023, 0.02)),
        ]
        for (metacarpal, y, knuckleZ, lengths) in rows {
            let base = metacarpal.rawValue
            p[metacarpal] = SIMD3(0, y * 0.4, -0.012)
            let knuckle = SIMD3<Float>(0, y, knuckleZ)
            p[HandJoint(rawValue: base + 1)!] = knuckle
            p[HandJoint(rawValue: base + 2)!] = knuckle + SIMD3(0, 0, -lengths.x)
            p[HandJoint(rawValue: base + 3)!] = knuckle + SIMD3(0, 0, -lengths.x - lengths.y)
            p[HandJoint(rawValue: base + 4)!] = knuckle + SIMD3(0, 0, -lengths.x - lengths.y - lengths.z)
        }
        return p
    }()

    static func rig() -> HandRig {
        let joints = HandJoint.allCases
        var local: [simd_float4x4] = []
        var inverseBind: [simd_float4x4] = []
        for j in joints {
            let at = restPositions[j]!
            let parentAt = j.parent.map { restPositions[$0]! } ?? .zero
            var m = matrix_identity_float4x4
            m.columns.3 = SIMD4(at - parentAt, 1)
            local.append(m)
            var inv = matrix_identity_float4x4
            inv.columns.3 = SIMD4(-at, 1)
            inverseBind.append(inv)
        }
        // The thumb bends about the axis across its own length and the
        // palm's normal, towards the palm.
        let thumbDir = simd_normalize(restPositions[.thumbTip]! - restPositions[.thumbKnuckle]!)
        let thumbAxis = simd_normalize(simd_cross(thumbDir, SIMD3<Float>(-1, 0, 0)))
        return HandRig(
            names: joints.map(\.name),
            parents: joints.map { $0.parent?.rawValue },
            restLocal: local,
            inverseBind: inverseBind,
            joint: Dictionary(uniqueKeysWithValues: joints.map { ($0, $0.rawValue) }),
            curlAxis: SIMD3(0, 1, 0),
            thumbAxis: thumbAxis,
            spreadAxis: SIMD3(1, 0, 0)
        )
    }

    /// Capsules from each joint to its child, a flattened palm and a short
    /// cuff over the wrist, each rigid on its joint (the palm and cuff on the
    /// wrist), with a little blend at the knuckles so a curl does not crack.
    static func mesh() -> SkinnedMesh {
        var mesh = SkinnedMesh()
        func capsule(from a: SIMD3<Float>, to b: SIMD3<Float>, radius: Float, joint: Int, blendTo parent: Int? = nil) {
            let axis = b - a
            let length = simd_length(axis)
            guard length > 1e-5 else { return }
            let z = axis / length
            let helper: SIMD3<Float> = abs(z.y) < 0.9 ? SIMD3(0, 1, 0) : SIMD3(1, 0, 0)
            let x = simd_normalize(simd_cross(helper, z))
            let y = simd_cross(z, x)
            let around = 12, rings = 6
            let start = UInt32(mesh.positions.count)
            // Rings: a hemisphere at each end and a straight middle.
            var profile: [(Float, Float, Float)] = []  // along, radial scale, normal's along part
            for r in 0...rings {
                let t = Float(r) / Float(rings) * (.pi / 2)
                profile.append((-radius * cos(t), sin(t), -cos(t)))
            }
            for r in 0...rings {
                let t = Float(r) / Float(rings) * (.pi / 2)
                profile.append((length + radius * sin(t), cos(t), sin(t)))
            }
            for (k, ring) in profile.enumerated() {
                for i in 0...around {
                    let phi = Float(i) / Float(around) * 2 * .pi
                    let radial = x * cos(phi) + y * sin(phi)
                    let n = simd_normalize(radial * ring.1 + z * ring.2)
                    mesh.positions.append(a + z * ring.0 + radial * (radius * ring.1))
                    mesh.normals.append(n)
                    mesh.tangents.append(SIMD4(simd_normalize(simd_cross(z, n) + 1e-6), 1))
                    mesh.uvs.append(SIMD2(Float(i) / Float(around), Float(k) / Float(profile.count - 1)))
                    // Near the base, partly on the parent, so the joint bends smoothly.
                    let along = max(0, min(1, (ring.0 + radius) / (2 * radius)))
                    if let parent, along < 1 {
                        let w = 0.5 + 0.5 * along
                        mesh.joints.append(SIMD4(UInt16(joint), UInt16(parent), 0, 0))
                        mesh.weights.append(SIMD4(w, 1 - w, 0, 0))
                    } else {
                        mesh.joints.append(SIMD4(UInt16(joint), 0, 0, 0))
                        mesh.weights.append(SIMD4(1, 0, 0, 0))
                    }
                }
            }
            let stride = UInt32(around + 1)
            for k in 0..<UInt32(profile.count - 1) {
                for i in 0..<UInt32(around) {
                    let a0 = start + k * stride + i, a1 = a0 + 1, b0 = a0 + stride, b1 = b0 + 1
                    mesh.indices += [a0, b0, a1, a1, b0, b1]
                }
            }
        }
        let p = restPositions
        let w = HandJoint.wrist.rawValue
        // Fingers: three phalanges each, thinner towards the tip.
        let radii: [Float] = [0.0105, 0.0098, 0.0088]
        for finger in 0..<5 {
            let bend = HandJoint.bending(finger: finger)
            for k in 0..<3 {
                let joint = bend[k]
                let child = HandJoint(rawValue: joint.rawValue + 1)!
                let scale: Float = finger == 0 ? 1.12 : finger == 4 ? 0.85 : 1
                capsule(from: p[joint]!, to: p[child]!, radius: radii[k] * scale, joint: joint.rawValue,
                        blendTo: joint.parent?.rawValue)
            }
        }
        // The palm: capsules from the wrist to each knuckle, wide enough to
        // close the gaps, all on the wrist.
        for knuckle in [HandJoint.indexFingerKnuckle, .middleFingerKnuckle, .ringFingerKnuckle, .littleFingerKnuckle] {
            capsule(from: SIMD3(0, p[knuckle]!.y * 0.55, -0.008), to: p[knuckle]!, radius: 0.0165, joint: w)
        }
        capsule(from: SIMD3(0, -0.012, -0.004), to: SIMD3(0, 0.016, -0.004), radius: 0.022, joint: w)
        capsule(from: SIMD3(0, 0.004, -0.004), to: p[.thumbKnuckle]!, radius: 0.016, joint: w)
        // A short cuff up the forearm.
        capsule(from: SIMD3(0, 0, 0.006), to: SIMD3(0, 0, 0.07), radius: 0.025, joint: HandJoint.forearmWrist.rawValue)
        return mesh
    }

    /// The left hand's mesh: the right's reflected in X, its triangles
    /// turned round so they still face out.
    static func mirrored(_ mesh: SkinnedMesh) -> SkinnedMesh {
        var out = mesh
        out.positions = mesh.positions.map { SIMD3(-$0.x, $0.y, $0.z) }
        out.normals = mesh.normals.map { SIMD3(-$0.x, $0.y, $0.z) }
        out.tangents = mesh.tangents.map { SIMD4(-$0.x, $0.y, $0.z, -$0.w) }
        for t in stride(from: 0, to: out.indices.count, by: 3) { out.indices.swapAt(t + 1, t + 2) }
        return out
    }
}

extension SIMD4 where Scalar == Float {
    var xyz: SIMD3<Float> { SIMD3(x, y, z) }
}
