/*
 * Test-only export: every pixel shader in the game's shader set,
 * paired with a vertex shader of the same family, through the runtime
 * chain into a Metal render pipeline, the way a draw links them. It
 * checks the claim draw.m rests on (unlinked stages pair by fixed output
 * locations) over the whole game, not only the pairs one menu run draws.
 *
 * A second export does the same across the fixed-function boundary:
 * every game pixel shader behind DXVK's fixed-function vertex shader,
 * and every game vertex shader in front of its fixed-function pixel
 * shader, the two mixed cases a draw can make.
 *
 * The input is build/s2-shaders/in: index.json and <model>/<hash>.dxso,
 * extracted from the user's own game files and never committed.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "mtl_d3d9.h"
#include "shader.h"

#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

/* The shader family a .vcs file belongs to, without its stage and model
   suffix, so psh/foo_ps20.vcs and vsh/foo_vs20.vcs pair. */
static NSString *family(NSString *vcs)
{
    NSString *b = vcs.lastPathComponent.stringByDeletingPathExtension;
    for (NSString *s in @[ @"_ps11", @"_ps14", @"_ps20", @"_vs11", @"_vs14", @"_vs20" ])
        if ([b hasSuffix:s]) return [b substringToIndex:b.length - s.length];
    return b;
}

typedef struct {
    _Atomic int pipelines, translated, cached, failed;
} pair_counts;

/* The highest texture index any MSL checked since the last reset names,
   arrays counted to their last element: what has to fit the smallest
   texture argument table (31 entries, indices 0 to 30). */
static _Atomic int s_max_slot = -1;

static void note_slots(const char *msl)
{
    int top = -1;
    for (const char *p = strstr(msl, "[[texture("); p; p = strstr(p + 1, "[[texture(")) {
        int n = atoi(p + 10), count = 1;
        const char *line = p;
        while (line > msl && line[-1] != '\n') line--;
        const char *arr = strstr(line, "array<texture");
        if (arr && arr < p) {
            /* array<texture2d<float>, 8>: the count follows the comma
               after the element type's closing bracket. */
            const char *close = strchr(arr, '>');
            const char *comma = close && close < p ? strchr(close, ',') : NULL;
            if (comma && comma < p) count = atoi(comma + 1);
        }
        if (n + count - 1 > top) top = n + count - 1;
    }
    int old = atomic_load(&s_max_slot);
    while (top > old && !atomic_compare_exchange_weak(&s_max_slot, &old, top)) {
    }
}

D9_EXPORT int metal_d3d9_selfcheck_max_slot(int reset)
{
    return reset ? atomic_exchange(&s_max_slot, -1) : atomic_load(&s_max_slot);
}

static id<MTLFunction> compile(id<MTLDevice> dev, mtl_sc *sc, const char *cache, pair_counts *n, NSString **why)
{
    char err[1024];
    int cached = 0;
    const char *msl = mtl_sc_msl(sc, cache, &cached, err, sizeof err);
    if (!msl) {
        *why = @(err);
        return nil;
    }
    atomic_fetch_add(cached ? &n->cached : &n->translated, 1);
    note_slots(msl);
    MTLCompileOptions *o = [MTLCompileOptions new];
    o.languageVersion = MTLLanguageVersion3_0;
    o.preserveInvariance = YES;
    NSError *e = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:@(msl) options:o error:&e];
    id<MTLFunction> fn = [lib newFunctionWithName:@"main0"];
    if (!fn) *why = e.localizedDescription;
    return fn;
}

static mtl_sc *load(NSString *dir, NSString *model, NSString *hash)
{
    NSData *d = [NSData dataWithContentsOfFile:[NSString stringWithFormat:@"%@/%@/%@.dxso", dir, model, hash]];
    if (!d) return NULL;
    char err[256];
    return mtl_sc_new(d.bytes, (uint32_t)d.length, err, sizeof err);
}

/* Returns the number of pairs that became pipelines; *tried is how many
   were attempted. limit 0 means every pixel shader. */
D9_EXPORT int metal_d3d9_pair_selfcheck(const char *in_dir, int limit, int *tried, int *translated, int *cached)
{
    __block pair_counts n;
    memset(&n, 0, sizeof n);
    __block _Atomic int attempted = 0;
    @autoreleasepool {
        NSString *dir = @(in_dir);
        NSData *jd = [NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"index.json"]];
        NSArray *ix = jd ? [NSJSONSerialization JSONObjectWithData:jd options:0 error:nil] : nil;
        if (!ix) return -1;
        NSMutableDictionary<NSString *, NSArray *> *vs_of = [NSMutableDictionary new];
        NSMutableArray *ps = [NSMutableArray new];
        NSMutableSet *seen = [NSMutableSet new];
        for (NSDictionary *e in ix) {
            NSString *model = e[@"model"], *hash = e[@"hash"];
            NSString *k = [NSString stringWithFormat:@"%@/%@", model, hash];
            if ([seen containsObject:k]) continue;
            [seen addObject:k];
            if ([model hasPrefix:@"vs"]) {
                NSString *f = family(e[@"vcs"]);
                if (!vs_of[f]) vs_of[f] = @[ model, hash ];
            } else {
                [ps addObject:@[ model, hash, family(e[@"vcs"]) ]];
            }
        }
        NSArray *fallback = vs_of[@"unlitgeneric"];
        if (!fallback) return -1;
        NSUInteger count = limit > 0 && (NSUInteger)limit < ps.count ? (NSUInteger)limit : ps.count;
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        const char *cache = getenv("RT_METAL_CACHE_DIR");
        dispatch_apply(count, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) {
            @autoreleasepool {
                NSArray *p = ps[i];
                NSArray *v = vs_of[p[2]] ? vs_of[p[2]] : fallback;
                atomic_fetch_add(&attempted, 1);
                mtl_sc *vsc = load(dir, v[0], v[1]), *psc = load(dir, p[0], p[1]);
                NSString *why = @"load or analysis failed";
                id<MTLFunction> vf = vsc ? compile(dev, vsc, cache, &n, &why) : nil;
                id<MTLFunction> pf = vf && psc ? compile(dev, psc, cache, &n, &why) : nil;
                id<MTLRenderPipelineState> pso = nil;
                if (vf && pf) {
                    /* Every input FLOAT4 from one stream, as a declaration
                       naming each semantic the vertex shader reads. */
                    const mtl_sc_info *vi = mtl_sc_get_info(vsc);
                    MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
                    for (uint32_t a = 0; a < vi->n_inputs; a++) {
                        vd.attributes[a].format = MTLVertexFormatFloat4;
                        vd.attributes[a].offset = 16 * a;
                        vd.attributes[a].bufferIndex = MTL_BUF_STREAM;
                    }
                    vd.layouts[MTL_BUF_STREAM].stride = 16 * (vi->n_inputs ? vi->n_inputs : 1);
                    MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
                    pd.vertexFunction = vf;
                    pd.fragmentFunction = pf;
                    pd.vertexDescriptor = vd;
                    pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
                    pd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
                    pd.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
                    NSError *e = nil;
                    pso = [dev newRenderPipelineStateWithDescriptor:pd error:&e];
                    if (!pso) why = e.localizedDescription;
                }
                if (pso) {
                    atomic_fetch_add(&n.pipelines, 1);
                } else if (atomic_fetch_add(&n.failed, 1) < 8) {
                    mtl_log("pair %s/%s + %s/%s: %s", [v[0] UTF8String], [v[1] UTF8String], [p[0] UTF8String],
                            [p[1] UTF8String], why.UTF8String);
                }
                if (vsc) mtl_sc_free(vsc);
                if (psc) mtl_sc_free(psc);
            }
        });
    }
    *tried = attempted;
    *translated = n.translated;
    *cached = n.cached;
    return n.pipelines;
}

static id<MTLFunction> compile_ff(id<MTLDevice> dev, int fragment, NSString **why)
{
    char err[1024];
    const char *msl = mtl_sc_ff_msl(fragment, err, sizeof err);
    if (!msl) {
        *why = @(err);
        return nil;
    }
    note_slots(msl);
    MTLCompileOptions *o = [MTLCompileOptions new];
    o.languageVersion = MTLLanguageVersion3_0;
    o.preserveInvariance = YES;
    NSError *e = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:@(msl) options:o error:&e];
    id<MTLFunction> fn = [lib newFunctionWithName:@"main0"];
    if (!fn) *why = e.localizedDescription;
    return fn;
}

/* Every unique game shader linked against the fixed-function stage on
   the other side. Returns the pipelines made; *tried counts attempts. */
D9_EXPORT int metal_d3d9_ff_pair_selfcheck(const char *in_dir, int *tried)
{
    __block _Atomic int made = 0, attempted = 0, failed = 0;
    @autoreleasepool {
        NSString *dir = @(in_dir);
        NSData *jd = [NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"index.json"]];
        NSArray *ix = jd ? [NSJSONSerialization JSONObjectWithData:jd options:0 error:nil] : nil;
        if (!ix) return -1;
        NSMutableArray *all = [NSMutableArray new];
        NSMutableSet *seen = [NSMutableSet new];
        for (NSDictionary *e in ix) {
            NSString *k = [NSString stringWithFormat:@"%@/%@", e[@"model"], e[@"hash"]];
            if ([seen containsObject:k]) continue;
            [seen addObject:k];
            [all addObject:@[ e[@"model"], e[@"hash"] ]];
        }
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        NSString *why = nil;
        id<MTLFunction> ffv = compile_ff(dev, 0, &why), fff = ffv ? compile_ff(dev, 1, &why) : nil;
        if (!ffv || !fff) {
            mtl_log("fixed-function stages did not compile: %s", why.UTF8String);
            return -1;
        }
        const char *cache = getenv("RT_METAL_CACHE_DIR");
        dispatch_apply(all.count, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) {
            @autoreleasepool {
                NSArray *p = all[i];
                int is_vs = [p[0] hasPrefix:@"vs"];
                atomic_fetch_add(&attempted, 1);
                pair_counts n;
                memset(&n, 0, sizeof n);
                NSString *err = @"load or analysis failed";
                mtl_sc *sc = load(dir, p[0], p[1]);
                id<MTLFunction> fn = sc ? compile(dev, sc, cache, &n, &err) : nil;
                id<MTLRenderPipelineState> pso = nil;
                if (fn) {
                    /* The vertex stage's inputs, every one FLOAT4 from one stream. */
                    mtl_sc_info ffi;
                    mtl_ff_input_signature(&ffi);
                    uint32_t ninputs = is_vs ? mtl_sc_get_info(sc)->n_inputs : ffi.n_inputs;
                    MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
                    for (uint32_t a = 0; a < ninputs; a++) {
                        vd.attributes[a].format = MTLVertexFormatFloat4;
                        vd.attributes[a].offset = 16 * a;
                        vd.attributes[a].bufferIndex = MTL_BUF_STREAM;
                    }
                    vd.layouts[MTL_BUF_STREAM].stride = 16 * (ninputs ? ninputs : 1);
                    MTLRenderPipelineDescriptor *pd = [MTLRenderPipelineDescriptor new];
                    pd.vertexFunction = is_vs ? fn : ffv;
                    pd.fragmentFunction = is_vs ? fff : fn;
                    pd.vertexDescriptor = vd;
                    pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
                    pd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
                    pd.stencilAttachmentPixelFormat = MTLPixelFormatDepth32Float_Stencil8;
                    NSError *e = nil;
                    pso = [dev newRenderPipelineStateWithDescriptor:pd error:&e];
                    if (!pso) err = e.localizedDescription;
                }
                if (pso) atomic_fetch_add(&made, 1);
                else if (atomic_fetch_add(&failed, 1) < 8)
                    mtl_log("fixed-function pair with %s/%s: %s", [p[0] UTF8String], [p[1] UTF8String], err.UTF8String);
                if (sc) mtl_sc_free(sc);
            }
        });
    }
    *tried = attempted;
    return made;
}
