import Foundation
import ModelIO
import simd

// A rigged hand from a file (USDZ, or anything ModelIO reads): its skinned
// mesh in this module's layout (`SkinnedMesh`), its skeleton as a
// `HandRig`, and its colour texture and material numbers. Joints are
// matched to visionOS's by the last part of their path, so a model rigged
// with those names poses as the stand-in does.

struct HandModel: @unchecked Sendable {
    var mesh: SkinnedMesh
    var rig: HandRig
    /// The base colour texture as ModelIO has it, for the renderer to load
    /// on its device; nil for a plain colour.
    var baseColourTexture: MDLTexture?
    /// Tangent space normals (+Y up, OpenGL's way) and roughness, linear.
    var normalTexture: MDLTexture?
    var roughnessTexture: MDLTexture?
    var baseColour: SIMD4<Float> = SIMD4(0.12, 0.09, 0.075, 1)
    var roughness: Float = 0.55
    var metallic: Float = 0
    /// Poses that came with the model (`poses.json` beside it).
    var poses: HandPoseFile?
    /// The model's own axes into this module's (fingers along -Z, the
    /// palm on -X for a right hand and +X for a left, the thumb up), so
    /// the places and fittings made for the stand-in fit it too.
    var fit = matrix_identity_float4x4
    var source: String
}

enum HandModelLoader {
    enum Failure: Error, CustomStringConvertible {
        case unreadable(String)
        case noMesh(String)
        case noSkeleton(String)

        var description: String {
            switch self {
            case .unreadable(let s): return "could not read \(s)"
            case .noMesh(let s): return "no mesh in \(s)"
            case .noSkeleton(let s): return "no skeleton bound in \(s)"
            }
        }
    }

    /// Loads the first skinned mesh in the file (every mesh bound to that
    /// skeleton, merged), and `poses.json` from the same folder if there
    /// is one. `fit` takes the model's axes into this module's; the
    /// default is a model made with the palm down (-Y), fingers along -Z
    /// and the right thumb on -X, turned a quarter about Z.
    static func load(url: URL, left: Bool, fit: simd_float4x4? = nil) throws -> HandModel {
        let fit = fit ?? simd_float4x4(simd_quatf(angle: (left ? 0.5 : -0.5) * .pi, axis: SIMD3(0, 0, 1)))
        guard MDLAsset.canImportFileExtension(url.pathExtension) else { throw Failure.unreadable(url.lastPathComponent) }
        let asset = MDLAsset(url: url, vertexDescriptor: nil, bufferAllocator: nil)
        asset.loadTextures()
        let meshes = (asset.childObjects(of: MDLMesh.self) as? [MDLMesh]) ?? []
        guard !meshes.isEmpty else { throw Failure.noMesh(url.lastPathComponent) }
        var bound: [(MDLMesh, MDLAnimationBindComponent)] = []
        for mesh in meshes {
            if let bind = bindComponent(of: mesh) { bound.append((mesh, bind)) }
        }
        guard let first = bound.first, let skeleton = first.1.skeleton else { throw Failure.noSkeleton(url.lastPathComponent) }
        var rig = makeRig(skeleton: skeleton, left: left, fit: fit)
        let order = rig.sortParentsFirst()
        var newIndex = [Int](repeating: 0, count: order.count)
        for (n, o) in order.enumerated() { newIndex[o] = n }

        var mesh = SkinnedMesh()
        var model = HandModel(mesh: mesh, rig: rig, baseColourTexture: nil, source: url.lastPathComponent)
        for (mdlMesh, bind) in bound where bind.skeleton === skeleton || bind.skeleton == nil {
            // The mesh's own joint list may be a subset of the skeleton's.
            let paths = bind.jointPaths ?? skeleton.jointPaths
            let toRig: [UInt16] = paths.map { path in
                let i = skeleton.jointPaths.firstIndex(of: path) ?? 0
                return UInt16(newIndex[i])
            }
            append(mdlMesh, geometryBind: floatMatrix(bind.geometryBindTransform), jointMap: toRig, into: &mesh, model: &model)
        }
        guard mesh.vertexCount > 0 else { throw Failure.noMesh(url.lastPathComponent) }
        model.mesh = mesh
        model.rig = rig
        let posesURL = url.deletingLastPathComponent().appendingPathComponent("poses.json")
        if let data = try? Data(contentsOf: posesURL) { model.poses = HandPoseFile(json: data, left: left) }
        model.fit = fit
        return model
    }

    private static func bindComponent(of object: MDLObject) -> MDLAnimationBindComponent? {
        var o: MDLObject? = object
        while let current = o {
            for c in current.components {
                if let bind = c as? MDLAnimationBindComponent { return bind }
            }
            o = current.parent
        }
        return nil
    }

    /// The rig from the skeleton: parents from the joint paths, rest
    /// transforms from its rest pose (else from its bind pose), inverse
    /// binds from its bind pose.
    private static func makeRig(skeleton: MDLSkeleton, left: Bool, fit: simd_float4x4) -> HandRig {
        let paths = skeleton.jointPaths
        let bind = skeleton.jointBindTransforms.float4x4Array
        var rest = skeleton.jointRestTransforms.float4x4Array
        let parents: [Int?] = paths.map { path in
            let parts = path.split(separator: "/")
            guard parts.count > 1 else { return nil }
            let parentPath = parts.dropLast().joined(separator: "/")
            return paths.firstIndex(of: parentPath) ?? paths.firstIndex(of: "/" + parentPath)
        }
        if rest.count != paths.count {
            // No rest pose: the bind pose, made local.
            rest = paths.indices.map { i in
                guard i < bind.count else { return matrix_identity_float4x4 }
                if let p = parents[i], p < bind.count { return bind[p].inverse * bind[i] }
                return bind[i]
            }
        }
        let names = paths.map { String($0.split(separator: "/").last ?? Substring($0)) }
        var joint: [HandJoint: Int] = [:]
        for (i, name) in names.enumerated() {
            if let j = HandJoint(name: name), joint[j] == nil { joint[j] = i }
        }
        let right = ProceduralHand.rig()
        var rig = HandRig(
            names: names,
            parents: parents,
            restLocal: rest,
            inverseBind: paths.indices.map { $0 < bind.count ? bind[$0].inverse : matrix_identity_float4x4 },
            joint: joint,
            curlAxis: right.curlAxis,
            thumbAxis: right.thumbAxis,
            spreadAxis: right.spreadAxis
        )
        let module = left ? right.mirrored() : right
        let back = simd_float3x3(columns: (fit.columns.0.xyz, fit.columns.1.xyz, fit.columns.2.xyz)).transpose
        rig.curlAxis = back * module.curlAxis
        rig.thumbAxis = back * module.thumbAxis
        rig.spreadAxis = back * module.spreadAxis
        rig.opposeAxis = back * module.opposeAxis
        return rig
    }

    /// One mesh's vertices into ours: laid out by ModelIO into position,
    /// normal, uv, joints and weights, put into model space through its
    /// geometry bind, its joints renumbered into the rig, its triangles
    /// appended; the first material's base colour taken.
    private static func append(
        _ mdlMesh: MDLMesh, geometryBind: simd_float4x4, jointMap: [UInt16], into mesh: inout SkinnedMesh,
        model: inout HandModel
    ) {
        if mdlMesh.vertexAttributeData(forAttributeNamed: MDLVertexAttributeNormal) == nil {
            mdlMesh.addNormals(withAttributeNamed: MDLVertexAttributeNormal, creaseThreshold: 0.5)
        }
        // Tangents with the bitangent's sign in w, from the uvs: the left
        // hand's are mirrored, so its sign is the other way.
        let hasUVs = mdlMesh.vertexAttributeData(forAttributeNamed: MDLVertexAttributeTextureCoordinate) != nil
        if hasUVs {
            mdlMesh.addOrthTanBasis(
                forTextureCoordinateAttributeNamed: MDLVertexAttributeTextureCoordinate,
                normalAttributeNamed: MDLVertexAttributeNormal, tangentAttributeNamed: MDLVertexAttributeTangent
            )
        }
        let layout = MDLVertexDescriptor()
        layout.attributes[0] = MDLVertexAttribute(name: MDLVertexAttributePosition, format: .float3, offset: 0, bufferIndex: 0)
        layout.attributes[1] = MDLVertexAttribute(name: MDLVertexAttributeNormal, format: .float3, offset: 12, bufferIndex: 0)
        layout.attributes[2] = MDLVertexAttribute(
            name: MDLVertexAttributeTextureCoordinate, format: .float2, offset: 24, bufferIndex: 0
        )
        layout.attributes[3] = MDLVertexAttribute(name: MDLVertexAttributeJointIndices, format: .uShort4, offset: 32, bufferIndex: 0)
        layout.attributes[4] = MDLVertexAttribute(name: MDLVertexAttributeJointWeights, format: .float4, offset: 40, bufferIndex: 0)
        layout.attributes[5] = MDLVertexAttribute(name: MDLVertexAttributeTangent, format: .float4, offset: 56, bufferIndex: 0)
        layout.layouts[0] = MDLVertexBufferLayout(stride: 72)
        mdlMesh.vertexDescriptor = layout
        guard let buffer = mdlMesh.vertexBuffers.first else { return }
        let map = buffer.map()
        let base = UInt32(mesh.vertexCount)
        let normalBind = simd_float3x3(
            columns: (geometryBind.columns.0.xyz, geometryBind.columns.1.xyz, geometryBind.columns.2.xyz)
        ).inverse.transpose
        let hasJoints = mdlMesh.vertexAttributeData(forAttributeNamed: MDLVertexAttributeJointIndices) != nil
        for v in 0..<mdlMesh.vertexCount {
            let p = map.bytes + v * 72
            let f = p.assumingMemoryBound(to: Float.self)
            let position = SIMD3(f[0], f[1], f[2])
            let normal = SIMD3(f[3], f[4], f[5])
            mesh.positions.append((geometryBind * SIMD4(position, 1)).xyz)
            let n = normalBind * normal
            mesh.normals.append(simd_length(n) > 0 ? simd_normalize(n) : SIMD3(0, 1, 0))
            mesh.uvs.append(SIMD2(f[6], 1 - f[7]))
            if hasUVs {
                // Flipping v turns the bitangent round: the sign goes with it.
                let t = normalBind * SIMD3(f[14], f[15], f[16])
                let tn = simd_length(t) > 0 ? simd_normalize(t) : SkinnedMesh.anyTangent(mesh.normals.last!).xyz
                mesh.tangents.append(SIMD4(tn, f[17] < 0 ? 1 : -1))
            } else {
                mesh.tangents.append(SkinnedMesh.anyTangent(mesh.normals.last!))
            }
            var joints = SIMD4<UInt16>(0, 0, 0, 0)
            var weights = SIMD4<Float>(1, 0, 0, 0)
            if hasJoints {
                let s = (p + 32).assumingMemoryBound(to: UInt16.self)
                let w = (p + 40).assumingMemoryBound(to: Float.self)
                weights = SIMD4(w[0], w[1], w[2], w[3])
                for k in 0..<4 {
                    let j = Int(s[k])
                    joints[k] = j < jointMap.count ? jointMap[j] : 0
                }
                let sum = weights.sum()
                weights = sum > 1e-6 ? weights / sum : SIMD4(1, 0, 0, 0)
            }
            mesh.joints.append(joints)
            mesh.weights.append(weights)
        }
        for case let submesh as MDLSubmesh in mdlMesh.submeshes ?? [] {
            guard submesh.geometryType == .triangles else { continue }
            let indexMap = submesh.indexBuffer.map()
            for i in 0..<submesh.indexCount {
                let index: UInt32
                switch submesh.indexType {
                case .uInt16: index = UInt32(indexMap.bytes.assumingMemoryBound(to: UInt16.self)[i])
                case .uInt8: index = UInt32(indexMap.bytes.assumingMemoryBound(to: UInt8.self)[i])
                default: index = indexMap.bytes.assumingMemoryBound(to: UInt32.self)[i]
                }
                mesh.indices.append(base + index)
            }
            if model.baseColourTexture == nil, let material = submesh.material {
                take(material, into: &model)
            }
        }
    }

    private static func take(_ material: MDLMaterial, into model: inout HandModel) {
        if let colour = material.property(with: .baseColor) {
            if colour.type == .texture, let texture = colour.textureSamplerValue?.texture {
                model.baseColourTexture = texture
                model.baseColour = SIMD4(1, 1, 1, 1)
            } else if colour.type == .float3 {
                model.baseColour = SIMD4(colour.float3Value, 1)
            } else if colour.type == .float4 {
                model.baseColour = colour.float4Value
            } else if colour.type == .color, let c = colour.color, let parts = c.components, c.numberOfComponents >= 3 {
                model.baseColour = SIMD4(Float(parts[0]), Float(parts[1]), Float(parts[2]), 1)
            }
        }
        if let r = material.property(with: .roughness) {
            if r.type == .float { model.roughness = r.floatValue }
            if r.type == .texture { model.roughnessTexture = r.textureSamplerValue?.texture }
        }
        if let n = material.property(with: .tangentSpaceNormal), n.type == .texture {
            model.normalTexture = n.textureSamplerValue?.texture
        }
        if let m = material.property(with: .metallic), m.type == .float { model.metallic = m.floatValue }
    }
}

private func floatMatrix(_ m: simd_double4x4) -> simd_float4x4 {
    simd_float4x4(
        SIMD4<Float>(m.columns.0), SIMD4<Float>(m.columns.1), SIMD4<Float>(m.columns.2), SIMD4<Float>(m.columns.3)
    )
}
