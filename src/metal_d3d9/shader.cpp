// The shader chain: D3D9 bytecode through DXVK's D3D9 front end to
// SPIR-V, then SPIRV-Cross to MSL. See shader.h.
//
// DXVK's converter class is defined in d3d9_shader.cpp with no header,
// so that file is compiled into this one. The DXVK side is fixed: the
// same assumed device options and D3D9 shader options every time,
// getCode with no linkage. That is the configuration
// all 4,520 game shaders were proven through.
//
// What this file adds is the Metal binding: explicit MSL indices for
// every DXVK set and binding (shader.h), one argument buffer for the
// sampler heap and plain bindings for everything else, and a disk cache
// of the MSL keyed by a hash of the bytecode and the chain's version.

#include "d3d9_shader.cpp"

#include <spirv_msl.hpp>

#include <CommonCrypto/CommonDigest.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <atomic>
#include <vector>

#include <d3d9_fixed_function_frag.h>
#include <d3d9_fixed_function_vert.h>

#include "shader.h"
#include "stereo.h"

namespace dxvk {
  // The D3D9 library defines this in d3d9_main.cpp, which is not linked.
  // Nothing is written unless DXVK logs, and mtl_sc_new points the log
  // path at none first.
  Logger Logger::s_instance("metal_d3d9.log");
}

using namespace dxvk;

namespace {

  // Bump when anything that changes the MSL changes: DXVK or SPIRV-Cross
  // pins, the options below, the binding map in shader.h.
  const char* const k_chain_tag = "dxvk-dd3a05bd_spvc-1.4.363.0_map2_stereo2";

  // As in the shader tests: a conformant driver without the heap, BDA,
  // sub-dword push data or 16-bit paths.
  DxvkShaderOptions device_options() {
    DxvkShaderOptions o = { };
    o.flags.set(DxvkShaderCompileFlag::SemanticIo);
    o.flags.set(DxvkShaderCompileFlag::LowerFtoI, DxvkShaderCompileFlag::LowerF32toF16);
    o.maxUniformBufferSize = 65536u;
    o.maxUniformBufferCount = -1;
    o.minStorageBufferAlignment = 16u;
    return o;
  }

  D3D9ShaderOptions d3d9_options() {
    D3D9ShaderOptions o = { };
    o.isSWVP = false;
    o.d3d9FloatEmulation = D3D9FloatEmulation::Enabled;
    o.forceSamplerTypeSpecConstants = false;
    return o;
  }

  void set_err(char* err, size_t len, const std::string& msg) {
    if (err && len)
      std::snprintf(err, len, "%s", msg.c_str());
  }

  void quiet_dxvk_log() {
    // DXVK opens a log file beside the executable on its first message
    // unless told otherwise; the run's stderr is enough. (Its level is
    // read once when the library loads, so only the path can be set here.)
    // Once only: shaders are created on the game thread while the render
    // thread translates others, and the environment is not for writing
    // while another thread may read it.
    static std::once_flag once;
    std::call_once(once, [] { setenv("DXVK_LOG_PATH", "none", 0); });
  }

  std::string read_file(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f)
      return std::string();
    std::string s;
    char b[65536];
    size_t n;
    while ((n = std::fread(b, 1, sizeof b, f)) > 0)
      s.append(b, n);
    std::fclose(f);
    return s;
  }

  // Written to a temporary name and renamed, so a run that dies midway
  // never leaves a truncated entry another run would trust.
  void write_file(const std::string& path, const std::string& data) {
    static std::atomic<uint32_t> seq { 0u };
    std::string tmp = path + ".tmp" + std::to_string(getpid()) + "." + std::to_string(seq++);
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f)
      return;
    bool ok = std::fwrite(data.data(), 1, data.size(), f) == data.size();
    ok = std::fclose(f) == 0 && ok;
    if (ok)
      std::rename(tmp.c_str(), path.c_str());
    else
      std::remove(tmp.c_str());
  }

  void bind_buffer(spirv_cross::CompilerMSL& c, spv::ExecutionModel m, uint32_t set, uint32_t binding, uint32_t index) {
    spirv_cross::MSLResourceBinding b;
    b.stage = m;
    b.desc_set = set;
    b.binding = binding;
    b.count = 1;
    b.msl_buffer = index;
    c.add_msl_resource_binding(b);
  }

  void bind_texture(spirv_cross::CompilerMSL& c, spv::ExecutionModel m, uint32_t set, uint32_t binding, uint32_t index) {
    spirv_cross::MSLResourceBinding b;
    b.stage = m;
    b.desc_set = set;
    b.binding = binding;
    b.count = 1;
    b.msl_texture = index;
    c.add_msl_resource_binding(b);
  }

  // SPIR-V to MSL with the backend's binding map. The texture sets are
  // DXVK's: set 2 for programmable shaders, the CBVs in set 1, the spec
  // data in set 7. The fixed-function ubershaders use other sets and go
  // through ff_msl with their own.
  uint32_t view_of(spv::Dim dim) {
    return dim == spv::Dim3D ? 1u : dim == spv::DimCube ? 2u : 0u;
  }

  // Every vertex stage ends with stereo's clip-space correction (stereo.h),
  // read from a buffer the draw binds rather than from a specialisation,
  // so one function and one pipeline serve the mono frame and both eyes;
  // the identity correction writes the position's own bits. It goes in
  // the MSL rather than the SPIR-V because SPIRV-Cross's push struct only
  // has the members DXVK's shader reads, and adding one there would move
  // DXVK's own offsets.
  void add_stereo(std::string& msl) {
    const std::string entry = "vertex main0_out main0(";
    size_t at = msl.find(entry);
    if (at == std::string::npos)
      throw std::runtime_error("stereo: no vertex entry point main0");
    size_t close = msl.find(")\n", at);
    if (close == std::string::npos)
      throw std::runtime_error("stereo: the entry point's parameters do not end");
    msl.insert(close, ", constant float4* mtl_stereo [[buffer(" + std::to_string(MTL_BUF_STEREO) + ")]]");
    const std::string ret = "return out;";
    const std::string fix = "out.gl_Position = mtl_stereo_apply(out.gl_Position, mtl_stereo);\n    ";
    int n = 0;
    for (size_t r = msl.find(ret, close); r != std::string::npos; r = msl.find(ret, r + fix.size() + ret.size())) {
      msl.insert(r, fix);
      n++;
    }
    if (!n)
      throw std::runtime_error("stereo: the entry point never returns its output");
    msl.insert(at,
               "// x' = s0.x x + s0.y w + s0.z, the same for y and z, and w' = s3.x w + s3.y w + s3.z.\n"
               "static inline __attribute__((always_inline)) float4 mtl_stereo_apply(float4 p, constant float4* s)\n"
               "{\n"
               "    return float4(s[0].x * p.x + s[0].y * p.w + s[0].z, s[1].x * p.y + s[1].y * p.w + s[1].z,\n"
               "                  s[2].x * p.z + s[2].y * p.w + s[2].z, s[3].x * p.w + s[3].y * p.w + s[3].z);\n"
               "}\n\n");
  }

  std::string spirv_to_msl(const uint32_t* words, size_t count, const mtl_sc_info& info) {
    bool fragment = info.is_ps != 0;
    spirv_cross::CompilerMSL c(words, count);
    auto model = fragment ? spv::ExecutionModelFragment : spv::ExecutionModelVertex;

    auto o = c.get_msl_options();
    o.platform = spirv_cross::CompilerMSL::Options::macOS;
    o.set_msl_version(3, 0);
    o.argument_buffers = true;
    o.argument_buffers_tier = spirv_cross::CompilerMSL::Options::ArgumentBuffersTier::Tier2;
    c.set_msl_options(o);

    // The sampler heap is an unsized array, which only a device argument
    // buffer can hold; everything else binds directly, so the draw path
    // needs no argument encoder and no useResource calls.
    c.set_argument_buffer_device_address_space(0, true);
    c.add_discrete_descriptor_set(1);
    c.add_discrete_descriptor_set(2);
    c.add_discrete_descriptor_set(7);

    bind_buffer(c, model, 0, spirv_cross::kArgumentBufferBinding, MTL_BUF_HEAP);
    {
      spirv_cross::MSLResourceBinding b;
      b.stage = model;
      b.desc_set = 0;
      b.binding = 0;
      b.count = 0;
      b.msl_sampler = 0;
      c.add_msl_resource_binding(b);
    }
    bind_buffer(c, model, spirv_cross::kPushConstDescSet, spirv_cross::kPushConstBinding, MTL_BUF_PUSH);
    bind_buffer(c, model, 7, 0, MTL_BUF_SPEC);
    for (uint32_t b = 0; b < 7; b++)
      bind_buffer(c, model, 1, b, MTL_BUF_CBV + b);

    // An untyped sampler is three variables on one binding. Each view
    // gets the texture index mtl_tex_layout gave it, which the draw path
    // binds from the same table. The layout is worked out from the
    // analysis alone, because MSL read back from the cache never comes
    // through here; an image the analysis did not foresee is an error,
    // not a guess, so the two can never disagree silently.
    auto res = c.get_shader_resources();
    for (auto& r : res.separate_images) {
      if (c.get_decoration(r.id, spv::DecorationDescriptorSet) != 2)
        continue;
      uint32_t binding = c.get_decoration(r.id, spv::DecorationBinding);
      uint32_t v = view_of(c.get_type(r.type_id).image.dim);
      uint32_t slot = binding < MTL_TEX_BINDINGS ? info.tex_slot[binding][v] : MTL_TEX_NONE;
      if (slot == MTL_TEX_NONE)
        throw std::runtime_error("image at binding " + std::to_string(binding) + " view " + std::to_string(v) +
                                 " has no texture index in the layout");
      // Renamed past every original binding, so no two views share one.
      c.set_decoration(r.id, spv::DecorationBinding, 128u + slot);
      bind_texture(c, model, 2, 128u + slot, slot);
    }

    std::string msl = c.compile();
    if (!fragment)
      add_stereo(msl);
    return msl;
  }

}

struct mtl_sc {
  std::vector<uint8_t> code;
  D3D9ShaderAnalysis analysis;
  mtl_sc_info info = { };
  char hash[33] = { };
  std::string msl;
};

extern "C" {

void mtl_tex_layout(mtl_sc_info* in) {
  std::memset(in->tex_slot, MTL_TEX_NONE, sizeof in->tex_slot);
  uint32_t n = 0;
  for (uint32_t v = 0; v < MTL_TEX_VIEWS; v++) {
    for (uint32_t b = 0; b < MTL_TEX_BINDINGS; b++) {
      if (!(in->sampler_mask & (1u << b)))
        continue;
      uint32_t t = in->view[b];
      bool typed = t == MTL_VIEW_2D || t == MTL_VIEW_3D || t == MTL_VIEW_CUBE;
      if (typed && t - MTL_VIEW_2D != v)
        continue;
      in->tex_slot[b][v] = uint8_t(n++);
    }
  }
  in->n_tex_slots = n;
}

mtl_sc* mtl_sc_new(const void* code, uint32_t bytes, char* err, size_t errlen) {
  quiet_dxvk_log();

  if (!code || bytes < 8u) {
    set_err(err, errlen, "bytecode too short");
    return nullptr;
  }

  auto s = new mtl_sc();
  s->code.assign(static_cast<const uint8_t*>(code), static_cast<const uint8_t*>(code) + bytes);

  uint8_t digest[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256(s->code.data(), CC_LONG(bytes), digest);
  for (int i = 0; i < 16; i++)
    std::snprintf(s->hash + 2 * i, 3, "%02x", digest[i]);

  try {
    s->analysis = D3D9ShaderAnalysis(dxbc_spv::util::ByteReader(s->code.data(), s->code.size()), false);
  } catch (const DxvkError& e) {
    set_err(err, errlen, "analysis: " + e.message());
    delete s;
    return nullptr;
  }

  uint32_t token = 0u;
  std::memcpy(&token, s->code.data(), 4u);

  auto& in = s->info;
  in.is_ps = (token >> 16) != 0xfffeu;
  in.major = (token >> 8) & 0xffu;
  in.minor = token & 0xffu;

  const auto& sig = s->analysis.GetInputSignature();
  in.n_inputs = std::min<uint32_t>(sig.size(), 16u);
  for (uint32_t i = 0; i < in.n_inputs; i++) {
    auto sem = sig.get(i);
    in.in_usage[i] = uint8_t(sem.usage);
    in.in_index[i] = uint8_t(sem.index);
  }

  in.sampler_mask = s->analysis.GetSamplerMask();
  for (uint32_t b = 0; b < MTL_TEX_BINDINGS; b++)
    in.view[b] = (in.sampler_mask & (1u << b)) ? uint32_t(s->analysis.GetImageViewType(b)) : 0u;
  mtl_tex_layout(&in);

  // Heap indices follow the stage's own push data (dxvk_shader_ir.cpp,
  // the D3D9 local push block): 32 for a pixel shader, 40 for a vertex
  // shader, whose cFSize and point size words come first.
  in.sampler_push_base = in.is_ps ? 32u : 40u;

  const auto* layout = s->analysis.GetConstantLayout();
  in.dynamic_floats = !in.is_ps && layout && layout->getLayout(D3D9ConstantType::Float).isDynamicallyIndexed();
  in.bool_mask = s->analysis.GetConstantsInfo().boolMask;
  if (!in.is_ps)
    in.pos_regs = mtl_stereo_scan_pos(reinterpret_cast<const uint32_t*>(s->code.data()), s->code.size() / 4u);
  return s;
}

void mtl_sc_free(mtl_sc* s) {
  delete s;
}

const mtl_sc_info* mtl_sc_get_info(const mtl_sc* s) {
  return &s->info;
}

const char* mtl_sc_hash(const mtl_sc* s) {
  return s->hash;
}

const char* mtl_sc_chain_tag(void) {
  return k_chain_tag;
}

char* mtl_sc_cached_msl(const char* cache_dir, const char* hash) {
  if (!cache_dir || !*cache_dir || !hash)
    return nullptr;
  std::string msl = read_file(std::string(cache_dir) + "/" + k_chain_tag + "/" + hash + ".metal");
  return msl.empty() ? nullptr : strdup(msl.c_str());
}

const char* mtl_sc_msl(mtl_sc* s, const char* cache_dir, int* cached, char* err, size_t errlen) {
  if (cached)
    *cached = 0;
  if (!s->msl.empty()) {
    if (cached)
      *cached = 1;
    return s->msl.c_str();
  }

  std::string path;
  if (cache_dir && *cache_dir) {
    std::string dir = std::string(cache_dir) + "/" + k_chain_tag;
    mkdir(dir.c_str(), 0755);
    path = dir + "/" + s->hash + ".metal";
    s->msl = read_file(path);
    if (!s->msl.empty()) {
      if (cached)
        *cached = 1;
      return s->msl.c_str();
    }
  }

  try {
    D3D9ShaderOptions opts = d3d9_options();

    auto hasher = dxbc_spv::util::md5::Hasher();
    hasher.update(s->code.data(), s->code.size());
    hasher.update(&opts, sizeof(opts));
    auto digest = hasher.finalize();

    VkShaderStageFlagBits stage = s->info.is_ps ? VK_SHADER_STAGE_FRAGMENT_BIT : VK_SHADER_STAGE_VERTEX_BIT;
    DxvkShaderHash key(stage, s->code.size(), digest.data.data(), digest.data.size());

    DxvkIrShaderCreateInfo info = { };
    info.options = device_options();
    info.flatShadingInputs = s->analysis.GetFlatShadingMask();

    Rc<D3D9ShaderConverter> converter = new D3D9ShaderConverter(key, opts, s->code.data(), s->analysis);
    Rc<DxvkIrShader> shader = new DxvkIrShader(info, std::move(converter));
    shader->compile();

    SpirvCodeBuffer spv = shader->getCode(nullptr, nullptr);
    s->msl = spirv_to_msl(spv.data(), spv.dwords(), s->info);
  } catch (const DxvkError& e) {
    set_err(err, errlen, "dxvk: " + e.message());
    return nullptr;
  } catch (const std::exception& e) {
    set_err(err, errlen, std::string("spirv-cross: ") + e.what());
    return nullptr;
  }

  if (!path.empty())
    write_file(path, s->msl);
  return s->msl.c_str();
}

// The sizes DXVK's UploadConstants gives the two buffers: the static CBV
// holds floats then ints when the floats are not dynamically indexed,
// ints alone when they are, and the dynamic buffer holds cF with one zero
// register past the count.
namespace {
struct const_sizes {
  uint32_t static_size, dynamic_size, dynamic_count, float_size, int_size;
  bool dynamic;
};

const_sizes constant_sizes(const mtl_sc* s, uint32_t changed_floats) {
  const_sizes z = { };
  const auto* layout = s->analysis.GetConstantLayout();
  if (!layout)
    return z;

  bool vs = !s->info.is_ps;
  if (vs)
    z.dynamic_count = layout->getLayout(D3D9ConstantType::Float).computeConstantCount(changed_floats);

  z.dynamic = vs && layout->getLayout(D3D9ConstantType::Float).isDynamicallyIndexed();
  auto sizes = layout->getAllocationSizes(z.dynamic_count);
  if (!vs)
    sizes.boolBufferSize = 0u;
  z.float_size = sizes.floatBufferSize;
  z.int_size = sizes.intBufferSize;

  z.static_size = sizes.boolBufferSize + sizes.intBufferSize;
  z.dynamic_size = sizes.floatBufferSize;
  if (!z.dynamic) {
    z.static_size += z.dynamic_size;
    z.dynamic_size = 0u;
  }
  if (z.static_size)
    z.static_size = std::min<uint32_t>(align(z.static_size, 16u), MTL_SC_MAX_CONST);
  if (z.dynamic_size)
    z.dynamic_size = std::min<uint32_t>(align(z.dynamic_size + uint32_t(sizeof(Vector4)), 16u), MTL_SC_MAX_CONST);
  return z;
}
}

void mtl_sc_constant_sizes(const mtl_sc* s, uint32_t changed_floats, uint32_t* static_bytes, uint32_t* dyn_bytes) {
  const_sizes z = constant_sizes(s, changed_floats);
  *static_bytes = z.static_size;
  *dyn_bytes = z.dynamic_size;
}

// DXVK's UploadConstants (d3d9_device.cpp), with the device's buffers
// replaced by the caller's, which are zeroed here over the sizes used.
void mtl_sc_constants(const mtl_sc* s, const float* f, const int32_t* i, const uint32_t* b, uint32_t changed_floats,
                      void* static_buf, uint32_t* static_bytes, void* dyn_buf, uint32_t* dyn_bytes, uint32_t* cf_size) {
  *static_bytes = 0;
  *dyn_bytes = 0;
  *cf_size = 0;

  const auto* layout = s->analysis.GetConstantLayout();
  if (!layout)
    return;

  bool vs = !s->info.is_ps;
  const_sizes z = constant_sizes(s, changed_floats);

  if (z.static_size) {
    std::memset(static_buf, 0, z.static_size);
    D3D9ConstantBufferCopyArgs a = { };
    a.flushNan = true;
    uint32_t off = 0u;
    if (!z.dynamic_size) {
      a.floatBuffer = static_cast<char*>(static_buf) + off;
      a.floatBufferSize = z.float_size;
      off += z.float_size;
    }
    a.intBuffer = static_cast<char*>(static_buf) + off;
    a.intBufferSize = z.int_size;
    off += z.int_size;
    a.boolBuffer = static_cast<char*>(static_buf) + off;
    a.boolBufferSize = z.static_size - off;
    a.constFloatApi = reinterpret_cast<const Vector4*>(f);
    a.constIntApi = reinterpret_cast<const Vector4i*>(i);
    a.constBoolApi = vs ? b : nullptr;
    layout->copyConstantData(a);
    *static_bytes = z.static_size;
  }

  if (z.dynamic_size) {
    std::memset(dyn_buf, 0, z.dynamic_size);
    D3D9ConstantBufferCopyArgs a = { };
    a.floatConstantCount = z.dynamic_count;
    a.flushNan = true;
    a.floatBuffer = dyn_buf;
    a.floatBufferSize = z.dynamic_size;
    a.constFloatApi = reinterpret_cast<const Vector4*>(f);
    layout->copyConstantData(a);
    *dyn_bytes = z.dynamic_size;
    *cf_size = z.dynamic_count;
  }
}

void mtl_ff_input_signature(mtl_sc_info* out) {
  // g_ffIsgn (d3d9_fixed_function.cpp) in D3DDECLUSAGE numbers: position,
  // normal, a second position and normal for tweening, eight texcoords,
  // two colours, fog, point size, blend weight and indices.
  static const uint8_t usage[MTL_MAX_INPUTS] = { 0, 3, 0, 3, 5, 5, 5, 5, 5, 5, 5, 5, 10, 10, 11, 4, 1, 2 };
  static const uint8_t index[MTL_MAX_INPUTS] = { 0, 0, 1, 1, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 0, 0, 0, 0 };
  *out = mtl_sc_info { };
  out->n_inputs = MTL_MAX_INPUTS;
  std::memcpy(out->in_usage, usage, sizeof usage);
  std::memcpy(out->in_index, index, sizeof index);
}

namespace {

  Matrix4 load_matrix(const float* m) {
    float rows[4][4];
    std::memcpy(rows, m, sizeof rows);
    return Matrix4(rows);
  }

}

static_assert(sizeof(D3D9FixedFunctionVS) <= MTL_FF_VS_BYTES, "ShaderData fits the size the draw path binds");
// Offsets the ubershader's scalar ShaderData block reads (as SPIRV-Cross
// lays it out): the material after eight 112-byte lights, then the
// packed bytes from DataPrimitives on.
static_assert(offsetof(D3D9FixedFunctionVS, Lights) == 864 && offsetof(D3D9FixedFunctionVS, Material) == 1760 &&
              offsetof(D3D9FixedFunctionVS, GlobalAmbient) == 1828 && offsetof(D3D9FixedFunctionVS, TexcoordIndices) == 1836 &&
              offsetof(D3D9FixedFunctionVS, VertexTexcoordDeclMask) == 1860 && offsetof(D3D9FixedFunctionVS, EmissiveSource) == 1881,
              "D3D9FixedFunctionVS matches the shader's ShaderData");
static_assert(sizeof(D3DLIGHT9) == 104 && sizeof(D3DMATERIAL9) == 68, "the bridge's light and material sizes");

// DXVK's UpdateFixedFunctionVS (d3d9_device.cpp), line for line where it
// computes something, with the device's state replaced by the caller's.
// DXVK's Matrix4 is used as DXVK uses it, so its operand order (view
// times world, which is world times view in D3D's row-vector terms) and
// its inverses carry over unchanged.
uint32_t mtl_ff_vs_data(const mtl_ff_vs_in* in, void* out, void* blend_out) {
  const uint32_t* rs = in->rs;
  auto tss = [in](uint32_t stage, uint32_t type) { return in->tss[stage * 33u + type]; };

  bool hasPositionT    = (in->decl_flags & MTL_DECL_POSITIONT) != 0;
  bool hasBlendWeight  = (in->decl_flags & MTL_DECL_BLENDWEIGHT) != 0;
  bool hasBlendIndices = (in->decl_flags & MTL_DECL_BLENDINDICES) != 0;

  bool indexedVertexBlend = hasBlendIndices && rs[D3DRS_INDEXEDVERTEXBLENDENABLE];
  uint32_t vertexBlendMode = D3D9FF_VertexBlendMode_Disabled;

  if (rs[D3DRS_VERTEXBLEND] != D3DVBF_DISABLE && !hasPositionT) {
    vertexBlendMode = rs[D3DRS_VERTEXBLEND] == D3DVBF_TWEENING
      ? D3D9FF_VertexBlendMode_Tween
      : D3D9FF_VertexBlendMode_Normal;

    if (rs[D3DRS_VERTEXBLEND] != D3DVBF_0WEIGHTS) {
      if (!hasBlendWeight)
        vertexBlendMode = D3D9FF_VertexBlendMode_Disabled;
    } else if (!indexedVertexBlend) {
      vertexBlendMode = D3D9FF_VertexBlendMode_Disabled;
    }
  }

  // The window to clip space mapping for pretransformed vertices. DXVK
  // computes it only with POSITIONT; it is cheap, so always.
  D3D9ViewportInfo vpInfo;
  vpInfo.inverseExtent = Vector4(
     2.0f / in->vp_w,
    -2.0f / in->vp_h,
    in->ztest ? 1.0f : 0.0f,
    1.0f);
  vpInfo.inverseOffset = Vector4(-in->vp_x, -in->vp_y, 0.0f, 0.0f);
  vpInfo.inverseOffset = vpInfo.inverseOffset * vpInfo.inverseExtent;
  vpInfo.inverseOffset = vpInfo.inverseOffset + Vector4(-1.0f, 1.0f, 0.0f, 0.0f);

  Matrix4 view = load_matrix(in->xf + 16u * MTL_XF_VIEW);
  Matrix4 world = load_matrix(in->xf + 16u * MTL_XF_WORLD0);

  auto WorldView    = view * world;
  auto NormalMatrix = tryInverse(WorldView).value_or(Matrix4(1.0f));
  auto Projection   = load_matrix(in->xf + 16u * MTL_XF_PROJ);

  // D3D9Light has no default constructor, so the block is built in
  // zeroed storage, the way DXVK's AllocTyped hands it out.
  alignas(16) uint8_t raw[sizeof(D3D9FixedFunctionVS)];
  std::memset(raw, 0, sizeof raw);
  auto& d = *reinterpret_cast<D3D9FixedFunctionVS*>(raw);
  d.WorldView     = WorldView;
  d.NormalMatrix  = NormalMatrix;
  d.InverseView   = transpose(inverse(view));
  d.Projection    = Projection;
  d.WorldViewProj = Projection * WorldView;

  for (uint32_t i = 0; i < d.TexcoordMatrices.size(); i++)
    d.TexcoordMatrices[i] = load_matrix(in->xf + 16u * (MTL_XF_TEX0 + i));

  d.ViewportInfo = vpInfo;

  uint32_t lightIdx = 0;
  for (uint32_t i = 0; i < in->nlights && lightIdx < caps::MaxEnabledLights; i++) {
    D3DLIGHT9 light;
    std::memcpy(&light, in->lights[i], sizeof light);

    // D3D9 accepts lights of invalid types but they light nothing.
    if (!light.Type || light.Type > D3DLIGHT_DIRECTIONAL)
      continue;

    d.Lights[lightIdx++] = D3D9Light(light, view);
  }

  std::memcpy(&d.Material, in->material, sizeof d.Material);
  d.GlobalAmbient = rs[D3DRS_AMBIENT];
  d.TweenFactor = bit::cast<float>(rs[D3DRS_TWEENFACTOR]);

  d.VertexHasPositionT = hasPositionT;
  d.VertexHasColor0    = (in->decl_flags & MTL_DECL_COLOR0) != 0;
  d.VertexHasColor1    = (in->decl_flags & MTL_DECL_COLOR1) != 0;
  d.VertexHasPointSize = (in->decl_flags & MTL_DECL_PSIZE) != 0;
  d.VertexHasFog       = (in->decl_flags & MTL_DECL_FOG) != 0;

  bool lighting    = rs[D3DRS_LIGHTING] != 0 && !hasPositionT;
  bool colorVertex = rs[D3DRS_COLORVERTEX] != 0;
  uint32_t mask    = (lighting && colorVertex)
                   ? (d.VertexHasColor0 ? D3DMCS_COLOR1 : D3DMCS_MATERIAL)
                   | (d.VertexHasColor1 ? D3DMCS_COLOR2 : D3DMCS_MATERIAL)
                   : 0;

  d.UseLighting      = lighting;
  d.NormalizeNormals = rs[D3DRS_NORMALIZENORMALS];
  d.LocalViewer      = rs[D3DRS_LOCALVIEWER] && lighting;
  d.RangeFog         = rs[D3DRS_RANGEFOGENABLE];

  d.DiffuseSource    = rs[D3DRS_DIFFUSEMATERIALSOURCE]  & mask;
  d.AmbientSource    = rs[D3DRS_AMBIENTMATERIALSOURCE]  & mask;
  d.SpecularSource   = rs[D3DRS_SPECULARMATERIALSOURCE] & mask;
  d.EmissiveSource   = rs[D3DRS_EMISSIVEMATERIALSOURCE] & mask;

  d.LightCount       = lighting ? lightIdx : 0;

  for (uint32_t i = 0; i < caps::MaxTextureBlendStages; i++) {
    uint32_t transformFlags = tss(i, D3DTSS_TEXTURETRANSFORMFLAGS) & ~(D3DTTFF_PROJECTED);
    uint32_t index          = tss(i, D3DTSS_TEXCOORDINDEX);
    uint32_t indexFlags     = (index & TCIMask) >> TCIOffset;

    transformFlags &= 0b111;
    index          &= 0b111;

    d.TexcoordTransformFlags[i] = transformFlags;
    d.TexcoordFlags[i]          = indexFlags;
    d.TexcoordIndices[i]        = index;
  }

  d.VertexTexcoordDeclMask = in->tc_mask;
  d.VertexBlendMode = uint8_t(vertexBlendMode);

  if (vertexBlendMode == D3D9FF_VertexBlendMode_Normal) {
    d.VertexBlendIndexed = indexedVertexBlend;
    d.VertexBlendCount   = rs[D3DRS_VERTEXBLEND] & 0xff;
  }

  d.VertexClipping = rs[D3DRS_CLIPPLANEENABLE] != 0;

  std::memset(out, 0, MTL_FF_VS_BYTES);
  std::memcpy(out, raw, sizeof raw);

  if (blend_out && vertexBlendMode == D3D9FF_VertexBlendMode_Normal) {
    // The hardware count: software vertex processing, which would index
    // up to 256, is never created by the bridge.
    auto* m = static_cast<Matrix4*>(blend_out);
    for (uint32_t i = 0; i < D3D9MaxVertexBlendTransformsHw; i++)
      m[i] = view * load_matrix(in->xf + 16u * (MTL_XF_WORLD0 + i));
  }

  return vertexBlendMode;
}

int mtl_ff_fog_use_z(const float* proj) {
  const auto m = load_matrix(proj);

  if (m[2][2] == 0.0f || m[2][2] == m[2][3])
    return 1;

  float wNear = m[3][3] - m[3][2] / m[2][2] * m[2][3];
  float wFar = (m[3][3] - m[3][2]) / (m[2][2] - m[2][3]) * m[2][3] + m[3][3];
  return wNear == 1.0f && wFar == 1.0f;
}

const char* mtl_sc_ff_msl(int which, char* err, size_t errlen) {
  static std::string out[2];
  // The pipeline warm threads and the render thread both ask for the
  // ubershaders. Once a string is filled it is never written again, so
  // the pointer returned stays valid after the lock is dropped.
  static std::mutex mu;
  std::lock_guard<std::mutex> hold(mu);
  if (which < 0 || which > 1)
    return nullptr;
  if (!out[which].empty())
    return out[which].c_str();

  try {
    const uint32_t* words = which ? d3d9_fixed_function_frag : d3d9_fixed_function_vert;
    size_t bytes = which ? sizeof(d3d9_fixed_function_frag) : sizeof(d3d9_fixed_function_vert);

    spirv_cross::CompilerMSL c(words, bytes / 4u);
    auto model = which ? spv::ExecutionModelFragment : spv::ExecutionModelVertex;
    auto o = c.get_msl_options();
    o.platform = spirv_cross::CompilerMSL::Options::macOS;
    o.set_msl_version(3, 0);
    o.argument_buffers = true;
    o.argument_buffers_tier = spirv_cross::CompilerMSL::Options::ArgumentBuffersTier::Tier2;
    c.set_msl_options(o);

    // The ubershaders' own sets: samplers 0, textures 1, CBVs 2, spec
    // data 3 (d3d9_fixed_function.cpp and its GLSL). As above, only the
    // sampler heap stays in an argument buffer.
    c.set_argument_buffer_device_address_space(0, true);
    c.add_discrete_descriptor_set(1);
    c.add_discrete_descriptor_set(2);
    c.add_discrete_descriptor_set(3);
    bind_buffer(c, model, 0, spirv_cross::kArgumentBufferBinding, MTL_BUF_HEAP);
    bind_buffer(c, model, spirv_cross::kPushConstDescSet, spirv_cross::kPushConstBinding, MTL_BUF_PUSH);
    bind_buffer(c, model, 3, 0, MTL_BUF_SPEC);
    for (uint32_t b = 0; b < 7; b++)
      bind_buffer(c, model, 2, b, MTL_BUF_CBV + b);
    {
      spirv_cross::MSLResourceBinding b;
      b.stage = model;
      b.desc_set = 0;
      b.binding = 0;
      b.count = 0;
      b.msl_sampler = 0;
      c.add_msl_resource_binding(b);
    }

    // The fragment ubershader declares its eight stage textures three
    // times on one binding (2D, cube and 3D arrays). Its stages are
    // untyped samplers 0 to 7, so mtl_tex_layout puts each view's eight
    // in a row; each array starts where its view's sampler 0 does, and
    // the draw path binds both kinds of pixel stage the same way.
    mtl_sc_info ff = { };
    ff.is_ps = 1;
    ff.sampler_mask = 0xFFu;
    mtl_tex_layout(&ff);
    auto res = c.get_shader_resources();
    for (auto& r : res.separate_images) {
      if (c.get_decoration(r.id, spv::DecorationDescriptorSet) != 1)
        continue;
      uint32_t base = ff.tex_slot[0][view_of(c.get_type(r.type_id).image.dim)];
      c.set_decoration(r.id, spv::DecorationBinding, 128u + base);
      spirv_cross::MSLResourceBinding b;
      b.stage = model;
      b.desc_set = 1;
      b.binding = 128u + base;
      b.count = 8;
      b.msl_texture = base;
      c.add_msl_resource_binding(b);
    }

    out[which] = c.compile();
    if (!which)
      add_stereo(out[which]);
  } catch (const std::exception& e) {
    set_err(err, errlen, std::string("spirv-cross: ") + e.what());
    return nullptr;
  }
  return out[which].c_str();
}

}
