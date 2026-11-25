import Foundation
import Metal
import MetalKit
import ModelIO
import simd

// Draws a skinned hand with Metal: the mesh uploaded once, skinned on the
// GPU (four joints a vertex), lit physically with one key light and a
// two colour sky, depth tested and depth writing. Optionally hidden where a
// scene the app already drew is nearer, from that scene's own depth in
// metres along the eye's forward (`SceneDepth`), which is how a hand goes
// behind a wall in a picture whose depth is not this pass's. The app owns
// the passes; this sets its state on an encoder it is given and draws.

/// A scene's depth for the hand to hide behind: per eye, metres along the
/// direction the image was drawn facing, 0 where nothing was drawn.
struct SceneDepth {
    var texture: MTLTexture
    var slice: Int
    /// Room space into the image's axes, the eye at the origin, -Z ahead.
    var imageFromOrigin: simd_float4x4
    /// The image's half angle tangents: left, right, down, up.
    var tangents: SIMD4<Float>
    /// How much nearer than the hand the scene must be to hide it, metres,
    /// and nearer than which the scene is ignored (what is held in the
    /// hand, drawn into that scene too).
    var bias: Float = 0.03
    var ignoreNearer: Float = 0
}

/// The light and surroundings a hand is lit by, in room space.
struct HandLighting {
    var light = simd_normalize(SIMD3<Float>(-0.42, 0.82, 0.39))
    var lightColour = SIMD3<Float>(1.0, 0.9, 0.78) * 1.6
    var sky = SIMD3<Float>(0.05, 0.05, 0.065)
    var ground = SIMD3<Float>(0.045, 0.035, 0.026)
    var horizon = SIMD3<Float>(0.55, 0.32, 0.12) * 0.25
}

/// A model on the GPU: its vertices and triangles, its colour texture.
final class HandMesh {
    let vertices: MTLBuffer
    let indices: MTLBuffer
    let indexCount: Int
    let jointCount: Int
    let texture: MTLTexture?
    let normalTexture: MTLTexture?
    let roughnessTexture: MTLTexture?
    /// The model's axes into the module's (`HandModel.fit`).
    let fit: simd_float4x4
    var baseColour: SIMD4<Float>
    var roughness: Float
    var metallic: Float

    init?(device: MTLDevice, mesh: SkinnedMesh, jointCount: Int, texture: MTLTexture? = nil,
          normalTexture: MTLTexture? = nil, roughnessTexture: MTLTexture? = nil, fit: simd_float4x4 = matrix_identity_float4x4,
          baseColour: SIMD4<Float> = SIMD4(0.12, 0.09, 0.075, 1), roughness: Float = 0.55, metallic: Float = 0) {
        let bytes = mesh.interleaved()
        guard !bytes.isEmpty, !mesh.indices.isEmpty,
              let v = device.makeBuffer(bytes: bytes, length: bytes.count, options: .storageModeShared),
              let i = device.makeBuffer(bytes: mesh.indices, length: mesh.indices.count * 4, options: .storageModeShared)
        else { return nil }
        v.label = "hand vertices"
        i.label = "hand indices"
        vertices = v
        indices = i
        indexCount = mesh.indices.count
        self.jointCount = jointCount
        self.texture = texture
        self.normalTexture = normalTexture
        self.roughnessTexture = roughnessTexture
        self.fit = fit
        self.baseColour = baseColour
        self.roughness = roughness
        self.metallic = metallic
    }

    /// A model loaded from a file, its textures loaded onto the device:
    /// the colour as sRGB, the normals and roughness as they are.
    convenience init?(device: MTLDevice, model: HandModel) {
        let loader = MTKTextureLoader(device: device)
        func load(_ source: MDLTexture?, srgb: Bool) -> MTLTexture? {
            guard let source else { return nil }
            return try? loader.newTexture(
                texture: source,
                options: [.SRGB: srgb, .generateMipmaps: true, .textureStorageMode: MTLStorageMode.private.rawValue]
            )
        }
        self.init(
            device: device, mesh: model.mesh, jointCount: model.rig.count, texture: load(model.baseColourTexture, srgb: true),
            normalTexture: load(model.normalTexture, srgb: false), roughnessTexture: load(model.roughnessTexture, srgb: false),
            fit: model.fit, baseColour: model.baseColour, roughness: model.roughness, metallic: model.metallic
        )
    }
}

final class HandRenderer {
    /// The most joints a palette carries: set as bytes, under the 4 KB
    /// limit for those.
    static let maxJoints = 48

    let device: MTLDevice
    private let pipeline: MTLRenderPipelineState
    private let depthState: MTLDepthStencilState
    private let white: MTLTexture
    private let noDepth: MTLTexture

    init(device: MTLDevice, colourFormat: MTLPixelFormat, depthFormat: MTLPixelFormat) throws {
        self.device = device
        let library = try device.makeLibrary(source: Self.source, options: nil)
        let d = MTLRenderPipelineDescriptor()
        d.label = "skinned hand"
        d.vertexFunction = library.makeFunction(name: "skinned_hand_vertex")
        d.fragmentFunction = library.makeFunction(name: "skinned_hand_fragment")
        d.colorAttachments[0].pixelFormat = colourFormat
        d.depthAttachmentPixelFormat = depthFormat
        pipeline = try device.makeRenderPipelineState(descriptor: d)
        let ds = MTLDepthStencilDescriptor()
        // Reverse Z: nearer is greater, as visionOS's projections give.
        ds.depthCompareFunction = .greater
        ds.isDepthWriteEnabled = true
        guard let state = device.makeDepthStencilState(descriptor: ds) else { throw HandRendererError.noState }
        depthState = state
        let t = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 1, height: 1, mipmapped: false)
        guard let w = device.makeTexture(descriptor: t) else { throw HandRendererError.noState }
        var one: UInt32 = 0xFFFF_FFFF
        w.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: &one, bytesPerRow: 4)
        white = w
        let z = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r16Float, width: 1, height: 1, mipmapped: false)
        z.textureType = .type2DArray
        z.arrayLength = 2
        guard let nd = device.makeTexture(descriptor: z) else { throw HandRendererError.noState }
        var zero: [UInt16] = [0]
        for s in 0..<2 {
            nd.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, slice: s, withBytes: &zero, bytesPerRow: 2,
                       bytesPerImage: 2)
        }
        noDepth = nd
    }

    enum HandRendererError: Error { case noState }

    /// Draws one hand: `model` places the mesh in the room (the hand's
    /// mirror for a left hand from a right mesh goes in it too), `palette`
    /// is the rig's (`HandRig.palette`). The encoder's pass must have a
    /// depth attachment of the format given at init.
    func draw(
        _ mesh: HandMesh, palette: [simd_float4x4], model: simd_float4x4, viewProjection: simd_float4x4,
        eye: SIMD3<Float>, lighting: HandLighting = HandLighting(), tint: SIMD4<Float> = SIMD4(1, 1, 1, 1),
        scene: SceneDepth? = nil, encoder: MTLRenderCommandEncoder
    ) {
        guard palette.count <= Self.maxJoints, !palette.isEmpty else { return }
        let model = model * mesh.fit
        var u = Uniforms(
            viewProjection: viewProjection,
            model: model,
            normalModel: Self.normalMatrix(model),
            imageFromOrigin: scene?.imageFromOrigin ?? matrix_identity_float4x4,
            eye: SIMD4(eye, 1),
            light: SIMD4(simd_normalize(lighting.light), 0),
            lightColour: SIMD4(lighting.lightColour, 0),
            sky: SIMD4(lighting.sky, 0),
            ground: SIMD4(lighting.ground, 0),
            horizon: SIMD4(lighting.horizon, 0),
            baseColour: mesh.baseColour * tint,
            material: SIMD4(mesh.roughness, mesh.metallic, mesh.texture != nil ? 1 : 0, mesh.normalTexture != nil ? 1 : 0),
            maps: SIMD4(mesh.roughnessTexture != nil ? 1 : 0, simd_determinant(model) < 0 ? -1 : 1, 0, 0),
            sceneTan: scene?.tangents ?? .zero,
            scene: SIMD4(scene != nil ? 1 : 0, Float(scene?.slice ?? 0), scene?.bias ?? 0, scene?.ignoreNearer ?? 0)
        )
        encoder.pushDebugGroup("skinned hand")
        encoder.setRenderPipelineState(pipeline)
        encoder.setDepthStencilState(depthState)
        // Both sides: a mirrored hand turns its triangles' winding round,
        // and a closed mesh behind the depth test costs little either way.
        encoder.setCullMode(.none)
        encoder.setVertexBuffer(mesh.vertices, offset: 0, index: 0)
        encoder.setVertexBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
        var p = palette
        encoder.setVertexBytes(&p, length: MemoryLayout<simd_float4x4>.stride * p.count, index: 2)
        encoder.setFragmentBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
        encoder.setFragmentTexture(mesh.texture ?? white, index: 0)
        encoder.setFragmentTexture(scene?.texture ?? noDepth, index: 1)
        encoder.setFragmentTexture(mesh.normalTexture ?? white, index: 2)
        encoder.setFragmentTexture(mesh.roughnessTexture ?? white, index: 3)
        encoder.drawIndexedPrimitives(
            type: .triangle, indexCount: mesh.indexCount, indexType: .uint32, indexBuffer: mesh.indices, indexBufferOffset: 0
        )
        encoder.popDebugGroup()
    }

    private static func normalMatrix(_ m: simd_float4x4) -> simd_float4x4 {
        let r = simd_float3x3(columns: (m.columns.0.xyz, m.columns.1.xyz, m.columns.2.xyz)).inverse.transpose
        return simd_float4x4(SIMD4(r.columns.0, 0), SIMD4(r.columns.1, 0), SIMD4(r.columns.2, 0), SIMD4(0, 0, 0, 1))
    }

    private struct Uniforms {
        var viewProjection: simd_float4x4
        var model: simd_float4x4
        var normalModel: simd_float4x4
        var imageFromOrigin: simd_float4x4
        var eye: SIMD4<Float>
        var light: SIMD4<Float>
        var lightColour: SIMD4<Float>
        var sky: SIMD4<Float>
        var ground: SIMD4<Float>
        var horizon: SIMD4<Float>
        var baseColour: SIMD4<Float>
        var material: SIMD4<Float>
        var maps: SIMD4<Float>
        var sceneTan: SIMD4<Float>
        var scene: SIMD4<Float>
    }

    static let source = """
        #include <metal_stdlib>
        using namespace metal;

        struct HandUniforms {
            float4x4 viewProjection;
            float4x4 model;
            float4x4 normalModel;
            float4x4 imageFromOrigin; // room into the scene image's axes, the eye at the origin
            float4 eye;
            float4 light;       // towards the light, room space
            float4 lightColour;
            float4 sky;
            float4 ground;
            float4 horizon;
            float4 baseColour;
            float4 material;    // x roughness, y metallic, z 1 with a colour texture, w 1 with a normal map
            float4 maps;        // x 1 with a roughness map, y -1 when the model is mirrored
            float4 sceneTan;    // the scene image's tangents: left, right, down, up
            float4 scene;       // x 1 to hide behind it, y its slice, z the bias, w nearer than which it is ignored
        };

        struct HandVertexIn {
            packed_float3 position;
            packed_float3 normal;
            float2 uv;
            ushort4 joints;
            packed_float4 weights;
            packed_float4 tangent;
        };

        struct HandOut {
            float4 position [[position]];
            float3 world;
            float3 normal;
            float4 tangent;
            float2 uv;
        };

        vertex HandOut skinned_hand_vertex(uint vid [[vertex_id]], const device HandVertexIn *v [[buffer(0)]],
                                           constant HandUniforms &u [[buffer(1)]],
                                           constant float4x4 *palette [[buffer(2)]]) {
            HandVertexIn in = v[vid];
            float4 w = float4(in.weights);
            float4x4 skin = palette[in.joints.x] * w.x + palette[in.joints.y] * w.y
                + palette[in.joints.z] * w.z + palette[in.joints.w] * w.w;
            float4 p = u.model * (skin * float4(float3(in.position), 1.0));
            float3 n = (u.normalModel * (skin * float4(float3(in.normal), 0.0))).xyz;
            float4 tg = float4(in.tangent);
            float3 t = (u.model * (skin * float4(tg.xyz, 0.0))).xyz;
            HandOut out;
            out.position = u.viewProjection * p;
            out.world = p.xyz;
            out.normal = n;
            out.tangent = float4(t, tg.w * u.maps.y);
            out.uv = in.uv;
            return out;
        }

        float3 hand_sky(constant HandUniforms &u, float3 d) {
            float3 c = d.y > 0.0 ? u.sky.rgb : mix(u.sky.rgb, u.ground.rgb, saturate(-d.y * 3.0));
            return c + u.horizon.rgb * exp(-abs(d.y) * 9.0);
        }

        // GGX with Smith's shadowing and Schlick's Fresnel, Lambert under it,
        // the sky for ambient and reflection, a soft rim against the dark.
        constexpr sampler ts(filter::linear, mip_filter::linear, address::repeat, max_anisotropy(8));

        fragment float4 skinned_hand_fragment(HandOut in [[stage_in]], constant HandUniforms &u [[buffer(1)]],
                                              texture2d<float> colour [[texture(0)]],
                                              texture2d_array<float> scene [[texture(1)]],
                                              texture2d<float> normals [[texture(2)]],
                                              texture2d<float> roughness [[texture(3)]]) {
            if (u.scene.x > 0.5) {
                float3 d = (u.imageFromOrigin * float4(in.world, 1.0)).xyz;
                if (d.z < -1e-3) {
                    float2 t = float2(d.x, d.y) / -d.z;
                    float4 tan = u.sceneTan;
                    float2 uv = float2((t.x + tan.x) / (tan.x + tan.y), (tan.w - t.y) / (tan.z + tan.w));
                    if (all(uv >= 0.0) && all(uv <= 1.0)) {
                        constexpr sampler s(filter::nearest, address::clamp_to_edge);
                        float at = scene.sample(s, uv, uint(u.scene.y)).r;
                        if (at > u.scene.w && at < -d.z - u.scene.z) discard_fragment();
                    }
                }
            }
            float4 base = u.baseColour * (u.material.z > 0.5 ? colour.sample(ts, in.uv) : float4(1.0));
            float3 n = normalize(in.normal);
            if (u.material.w > 0.5) {
                // Tangent space, +Y up as OpenGL has it; the bitangent's
                // sign from the tangent's w, the mirrored hand's included.
                float3 t = in.tangent.xyz - n * dot(n, in.tangent.xyz);
                if (length(t) > 1e-5) {
                    t = normalize(t);
                    float3 b = cross(n, t) * (in.tangent.w < 0.0 ? -1.0 : 1.0);
                    float3 m = normals.sample(ts, in.uv).xyz * 2.0 - 1.0;
                    n = normalize(t * m.x + b * m.y + n * m.z);
                }
            }
            float3 v = normalize(u.eye.xyz - in.world);
            float3 l = normalize(u.light.xyz);
            float3 h = normalize(l + v);
            float rough = clamp(u.material.x * (u.maps.x > 0.5 ? roughness.sample(ts, in.uv).g : 1.0), 0.04, 1.0);
            float a = rough * rough;
            float ndl = saturate(dot(n, l)), ndv = max(dot(n, v), 1e-3), ndh = saturate(dot(n, h)), vdh = saturate(dot(v, h));
            float a2 = a * a;
            float dd = ndh * ndh * (a2 - 1.0) + 1.0;
            float D = a2 / (M_PI_F * dd * dd);
            float k = (rough + 1.0) * (rough + 1.0) / 8.0;
            float G = (ndl / (ndl * (1.0 - k) + k)) * (ndv / (ndv * (1.0 - k) + k));
            float3 f0 = mix(float3(0.04), base.rgb, u.material.y);
            float3 F = f0 + (1.0 - f0) * pow(1.0 - vdh, 5.0);
            float3 spec = D * G * F / max(4.0 * ndl * ndv, 1e-3);
            float3 diffuse = (1.0 - F) * (1.0 - u.material.y) * base.rgb / M_PI_F;
            float3 c = (diffuse + spec) * u.lightColour.rgb * ndl;
            // Ambient: the sky over the hemisphere the normal faces, and its
            // reflection, dimmer for a rougher surface.
            float3 ambient = mix(hand_sky(u, -n), hand_sky(u, n), 0.75) * 3.0;
            float3 fAmb = f0 + (max(float3(1.0 - rough), f0) - f0) * pow(1.0 - ndv, 5.0);
            c += ambient * base.rgb * (1.0 - u.material.y) + hand_sky(u, reflect(-v, n)) * fAmb * (1.0 - rough * 0.7) * 3.0;
            c += pow(1.0 - ndv, 3.0) * 0.05 * u.lightColour.rgb;
            return float4(c, 1.0);
        }
        """
}
