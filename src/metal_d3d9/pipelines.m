/*
 * Stage functions and render pipelines that outlive a run. See
 * pipelines.h for the files and the lookup rules.
 *
 * Two tables, stage key to MTLFunction and record to pipeline state,
 * hold one slot per entry: busy while one thread builds it, then ready
 * or failed. Whoever finds no slot claims it and builds; whoever finds a
 * busy one waits on the one condition variable (or, in skip mode, goes
 * on without it). The warm queue and the replay thread claim the same
 * way, so nothing is ever built twice and the replay thread only waits
 * for work already under way.
 *
 * The archive is only read while the run lasts. Every pipeline made
 * keeps its descriptor, and a new archive is filled from them when the
 * device closes, with nothing else running (save_cache says why it is
 * new rather than added to). Metal does not document an archive as safe
 * to add to while other threads create pipelines from it, and the close
 * is the one moment that question does not arise.
 *
 * The game never releases its device when it quits, so the close cannot
 * be the only moment: mtl_pipes_finish does the same save and summary
 * without freeing, for the process exit hook. Both go through it, it
 * runs once per state of the cache (a second call saves only what was
 * made since the first), and the summary is logged once.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "pipelines.h"
#include "mtl_d3d9.h"
#include "shader.h"

#include <CommonCrypto/CommonDigest.h>
#include <mach/mach_time.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define KEYS_MAGIC "vtmb-metal-pipelines 1"
#define WARM_WIDTH 3 /* pipelines the warm queue builds at once: enough to keep Metal's compiler busy, few enough
                        that the game thread keeps its cores while the menu loads */
#define REC_HEAD (2 * MTL_PIPES_STAGE_KEY)

enum { SLOT_BUSY, SLOT_READY, SLOT_FAILED };

@interface MtlPipeSlot : NSObject {
@public
    int state;
    id obj;
}
@end

@implementation MtlPipeSlot
@end

@interface MtlPipes : NSObject {
@public
    pthread_mutex_t mu;
    pthread_cond_t cv;
    id<MTLDevice> dev;
    NSString *cache_dir; /* nil: nothing persists */
    NSString *dir;       /* cache_dir/<chain tag> */
    NSString *ident;
    size_t desc_bytes;
    mtl_pipes_describe describe;
    id<MTLBinaryArchive> archive;
    int opened, skip;
    int fail;      /* test: every pipeline build fails */
    int open_late; /* the archive did not open in time; no build waits for it again */
    int rebuilt;  /* the files on disk were thrown away at open, so they must be written even with nothing new */
    int reported; /* the exit summary is logged */
    pthread_mutex_t finish_mu; /* one finish at a time: the exit hook and a release may race */
    _Atomic int cancel;
    NSMutableDictionary<NSString *, MtlPipeSlot *> *fns;
    NSMutableDictionary<NSData *, MtlPipeSlot *> *psos;
    NSArray<NSData *> *recorded;
    NSMutableOrderedSet<NSData *> *used;
    NSMutableArray<MTLRenderPipelineDescriptor *> *built; /* every pipeline made this run, for the next archive */
    NSUInteger cold_built;
    dispatch_queue_t driver, workers;
    dispatch_group_t all, warm_jobs;
    dispatch_semaphore_t width;
    uint64_t st[MTL_PIPES_STAT_N];
}
@end

@implementation MtlPipes
- (void)dealloc
{
    pthread_mutex_destroy(&finish_mu);
    pthread_cond_destroy(&cv);
    pthread_mutex_destroy(&mu);
}
@end

#define P(p) ((__bridge MtlPipes *)(void *)(p))

static double now_ms(void)
{
    static mach_timebase_info_data_t tb;
    if (!tb.denom) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom / 1e6;
}

static NSString *sha256_hex(const void *bytes, size_t n)
{
    uint8_t h[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(bytes, (CC_LONG)n, h);
    NSMutableString *s = [NSMutableString stringWithCapacity:2 * sizeof h];
    for (size_t i = 0; i < sizeof h; i++) [s appendFormat:@"%02x", h[i]];
    return s;
}

static NSData *make_record(MtlPipes *p, const char *vs, const char *ps, const void *desc)
{
    NSMutableData *r = [NSMutableData dataWithLength:REC_HEAD + p->desc_bytes];
    uint8_t *b = r.mutableBytes;
    strncpy((char *)b, vs, MTL_PIPES_STAGE_KEY);
    strncpy((char *)b + MTL_PIPES_STAGE_KEY, ps, MTL_PIPES_STAGE_KEY);
    memcpy(b + REC_HEAD, desc, p->desc_bytes);
    return r;
}

static NSString *record_stage(NSData *r, int fragment)
{
    const char *k = (const char *)r.bytes + (fragment ? MTL_PIPES_STAGE_KEY : 0);
    return [[NSString alloc] initWithBytes:k length:strnlen(k, MTL_PIPES_STAGE_KEY) encoding:NSASCIIStringEncoding];
}

id<MTLFunction> mtl_pipes_compile(id<MTLDevice> dev, NSString *msl, NSError **e)
{
    MTLCompileOptions *o = [MTLCompileOptions new];
    o.languageVersion = MTLLanguageVersion3_0;
    o.preserveInvariance = YES;
    /* DXVK's float emulation (0 * inf = 0 multiplies) relies on IEEE
       behaviour that fast math may fold away. */
    if (@available(macOS 15.0, *)) {
        o.mathMode = MTLMathModeSafe;
    } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        o.fastMathEnabled = NO;
#pragma clang diagnostic pop
    }
    id<MTLLibrary> lib = [dev newLibraryWithSource:msl options:o error:e];
    return [lib newFunctionWithName:@"main0"];
}

/* The files */

static NSString *archive_path(MtlPipes *p) { return [p->dir stringByAppendingPathComponent:@"pipelines.bin"]; }
static NSString *keys_path(MtlPipes *p) { return [p->dir stringByAppendingPathComponent:@"pipelines.keys"]; }

/* The records of a keys file that matches this run and the archive
   beside it, or nil with the reason and whether it was a mismatch (a
   file written for another build, GPU or OS) rather than damage. */
static NSArray<NSData *> *parse_keys(MtlPipes *p, NSData *kd, NSData *ad, NSString **why, int *mismatch)
{
    *mismatch = 0;
    if (!kd || !ad) {
        *why = kd ? @"the keys have no archive" : @"the archive has no keys";
        return nil;
    }
    const uint8_t *b = kd.bytes;
    size_t n = kd.length;
    if (n < CC_SHA256_DIGEST_LENGTH + sizeof KEYS_MAGIC) {
        *why = @"the keys file is truncated";
        return nil;
    }
    size_t body = n - CC_SHA256_DIGEST_LENGTH;
    uint8_t h[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(b, (CC_LONG)body, h);
    if (memcmp(h, b + body, sizeof h)) {
        *why = @"the keys file fails its checksum";
        return nil;
    }
    /* Four text lines, then the records. */
    size_t pos = 0;
    NSString *line[4];
    for (int i = 0; i < 4; i++) {
        const uint8_t *nl = memchr(b + pos, '\n', body - pos);
        if (!nl) {
            *why = @"the keys header is cut short";
            return nil;
        }
        line[i] = [[NSString alloc] initWithBytes:b + pos length:(size_t)(nl - (b + pos)) encoding:NSUTF8StringEncoding];
        pos = (size_t)(nl - b) + 1;
        if (!line[i]) {
            *why = @"the keys header is not text";
            return nil;
        }
    }
    if (![line[0] isEqualToString:@KEYS_MAGIC]) {
        *mismatch = [line[0] hasPrefix:@"vtmb-metal-pipelines "];
        *why = [NSString stringWithFormat:@"the keys file format is \"%@\"", line[0]];
        return nil;
    }
    NSString *want = [@"ident " stringByAppendingString:p->ident];
    if (![line[1] isEqualToString:want]) {
        *mismatch = 1;
        *why = [NSString stringWithFormat:@"made for another build, GPU or OS (%@)", line[1]];
        return nil;
    }
    unsigned long long asize = 0, count = 0, rbytes = 0;
    char hex[65] = {0};
    if (sscanf(line[2].UTF8String, "archive %llu %64s", &asize, hex) != 2 ||
        sscanf(line[3].UTF8String, "records %llu %llu", &count, &rbytes) != 2) {
        *why = @"the keys header does not parse";
        return nil;
    }
    if (asize != ad.length || ![@(hex) isEqualToString:sha256_hex(ad.bytes, ad.length)]) {
        *why = @"the archive is not the one its keys were written with";
        return nil;
    }
    if (rbytes != REC_HEAD + p->desc_bytes || count * rbytes != body - pos) {
        *why = @"the records do not fit the keys file";
        return nil;
    }
    NSMutableArray *recs = [NSMutableArray arrayWithCapacity:(NSUInteger)count];
    for (unsigned long long i = 0; i < count; i++)
        [recs addObject:[NSData dataWithBytes:b + pos + i * rbytes length:(NSUInteger)rbytes]];
    return recs;
}

static void open_cache(MtlPipes *p)
{
    double t0 = now_ms();
    NSFileManager *fm = [NSFileManager defaultManager];
    [fm createDirectoryAtPath:p->dir withIntermediateDirectories:YES attributes:nil error:nil];
    NSString *ap = archive_path(p), *kp = keys_path(p);
    NSData *kd = [NSData dataWithContentsOfFile:kp];
    NSData *ad = [NSData dataWithContentsOfFile:ap options:NSDataReadingMappedIfSafe error:nil];
    uint64_t state = 1;
    NSArray<NSData *> *recs = nil;
    id<MTLBinaryArchive> a = nil;
    if (kd || ad) {
        NSString *why = nil;
        int mismatch = 0;
        recs = parse_keys(p, kd, ad, &why, &mismatch);
        if (recs) {
            MTLBinaryArchiveDescriptor *d = [MTLBinaryArchiveDescriptor new];
            d.url = [NSURL fileURLWithPath:ap];
            NSError *e = nil;
            a = [p->dev newBinaryArchiveWithDescriptor:d error:&e];
            if (!a) {
                why = [NSString stringWithFormat:@"Metal refuses the archive (%@)", e.localizedDescription];
                recs = nil;
            }
        }
        if (!recs) {
            state = mismatch ? 4 : 3;
            mtl_log("pipelines: %s %s: %s; deleted, rebuilding from nothing", mismatch ? "mismatched" : "damaged",
                    p->dir.UTF8String, why.UTF8String);
            [fm removeItemAtPath:ap error:nil];
            [fm removeItemAtPath:kp error:nil];
        } else {
            state = 2;
        }
    }
    if (!a) {
        NSError *e = nil;
        a = [p->dev newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new] error:&e];
        if (!a)
            mtl_log("pipelines: no binary archive (%s); every pipeline is compiled each run",
                    e.localizedDescription.UTF8String);
    }
    ad = nil; /* the mapping is not needed past the hash */
    pthread_mutex_lock(&p->mu);
    p->archive = a;
    p->recorded = recs ? recs : @[];
    p->st[MTL_PIPES_STAT_OPEN] = state;
    p->st[MTL_PIPES_STAT_RECORDED] = p->recorded.count;
    p->rebuilt = state >= 3;
    p->opened = 1;
    pthread_cond_broadcast(&p->cv);
    pthread_mutex_unlock(&p->mu);
    static const char *const k_state[] = {"", "new", "loaded", "rebuilt", "rebuilt"};
    mtl_log("pipelines: %s %s, %lu recorded pipelines to warm, opened in %.1f ms", k_state[state],
            archive_path(p).UTF8String, (unsigned long)p->recorded.count, now_ms() - t0);
}

/* Called with nothing else running: the warm queue is drained.

   The archive is written new each time, from every pipeline made this
   run, rather than added to. Adding to an archive loaded from a file
   drops entries it already holds when a new pipeline shares a stage
   with them (seen with DXVK's fixed-function vertex shader in front of
   two different pixel shaders: each save lost the other's pipeline, so
   one of them was compiled cold on every run), while an archive filled
   in one go keeps them all. The descriptors of pipelines that came from
   the old archive still name it, so Metal copies those rather than
   compiling them again. A recorded pipeline the warm queue did not reach
   keeps its record, and is built again off the critical path next run. */
static void save_cache(MtlPipes *p)
{
    /* A snapshot under the lock: at process exit the device is still
       live, and nothing guarantees no other thread asks for a pipeline
       while this runs. Whatever arrives after it is saved by the next
       finish, if there is one. */
    pthread_mutex_lock(&p->mu);
    NSMutableOrderedSet<NSData *> *recs = [NSMutableOrderedSet orderedSetWithArray:p->recorded];
    NSUInteger before = recs.count;
    /* A pipeline that failed is not recorded: the next run would only
       spend its warm queue failing it again. */
    for (NSData *r in p->used) {
        MtlPipeSlot *slot = p->psos[r];
        if (!slot || slot->state != SLOT_FAILED) [recs addObject:r];
    }
    NSArray<MTLRenderPipelineDescriptor *> *built = [p->built copy];
    NSUInteger cold = p->cold_built;
    int rebuilt = p->rebuilt;
    pthread_mutex_unlock(&p->mu);
    if (!cold && recs.count == before && !rebuilt) return;
    static pthread_mutex_t save_mu = PTHREAD_MUTEX_INITIALIZER; /* two devices closing at once in one process */
    pthread_mutex_lock(&save_mu);
    double t0 = now_ms();
    NSString *ap = archive_path(p), *kp = keys_path(p);
    /* Records another device saved since this one opened (the headless
       test keeps two alive at once) are kept, so the next run still
       warms them. */
    NSString *why = nil;
    int mismatch = 0;
    NSData *ad = [NSData dataWithContentsOfFile:ap options:NSDataReadingMappedIfSafe error:nil];
    NSArray<NSData *> *disk = parse_keys(p, [NSData dataWithContentsOfFile:kp], ad, &why, &mismatch);
    ad = nil;
    if (disk) {
        NSMutableOrderedSet<NSData *> *merged = [NSMutableOrderedSet orderedSetWithArray:disk];
        [merged unionOrderedSet:recs];
        recs = merged;
    }
    NSError *e = nil;
    id<MTLBinaryArchive> fresh = [p->dev newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new] error:&e];
    if (!fresh) {
        mtl_log("pipelines: archive not written (%s)", e.localizedDescription.UTF8String);
        pthread_mutex_unlock(&save_mu);
        return;
    }
    NSUInteger added = 0, refused = 0;
    for (MTLRenderPipelineDescriptor *pd in built) {
        e = nil;
        if ([fresh addRenderPipelineFunctionsWithDescriptor:pd error:&e]) {
            added++;
        } else if (refused++ < 4) {
            mtl_log("pipelines: the archive refused a pipeline: %s", e.localizedDescription.UTF8String);
        }
    }
    NSString *tmp = [ap stringByAppendingFormat:@".tmp%d", getpid()];
    e = nil;
    BOOL ok = [fresh serializeToURL:[NSURL fileURLWithPath:tmp] error:&e];
    ad = ok ? [NSData dataWithContentsOfFile:tmp options:NSDataReadingMappedIfSafe error:nil] : nil;
    if (!ad) {
        mtl_log("pipelines: archive not written (%s)", ok ? "unreadable after writing" : e.localizedDescription.UTF8String);
        unlink(tmp.UTF8String);
        pthread_mutex_unlock(&save_mu);
        return;
    }
    size_t rbytes = REC_HEAD + p->desc_bytes;
    NSMutableData *kd = [NSMutableData new];
    NSString *head = [NSString stringWithFormat:@"%s\nident %@\narchive %lu %@\nrecords %lu %zu\n", KEYS_MAGIC, p->ident,
                                                (unsigned long)ad.length, sha256_hex(ad.bytes, ad.length),
                                                (unsigned long)recs.count, rbytes];
    [kd appendData:[head dataUsingEncoding:NSUTF8StringEncoding]];
    for (NSData *r in recs) [kd appendData:r];
    uint8_t h[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(kd.bytes, (CC_LONG)kd.length, h);
    [kd appendBytes:h length:sizeof h];
    ad = nil;
    /* The archive lands first: keys that name an archive which is not
       there yet read as damage next run, never as a match. */
    ok = rename(tmp.UTF8String, ap.UTF8String) == 0 && [kd writeToFile:kp atomically:YES];
    mtl_log("pipelines: %s %s: %lu pipelines (%lu new), %lu recorded, in %.0f ms", ok ? "saved" : "could not save",
            ap.UTF8String, (unsigned long)added, (unsigned long)cold, (unsigned long)recs.count, now_ms() - t0);
    if (ok) {
        /* What is on disk now is the baseline: a later finish (the exit
           hook, then the release) writes again only for pipelines made
           after this one. */
        pthread_mutex_lock(&p->mu);
        p->recorded = [recs.array copy]; /* the array of a mutable set is a live view of it */
        p->cold_built -= cold;
        p->rebuilt = 0;
        pthread_mutex_unlock(&p->mu);
    }
    pthread_mutex_unlock(&save_mu);
}

/* Building */

/* A pipeline from the archive if it has one, else compiled; cold
   descriptors are kept for the archive. Waits for the archive to open,
   which takes milliseconds, but not past OPEN_WAIT_S: an unbounded wait
   tied every draw to the warm queue's thread, and a Metal call that
   never returned there would have stopped the frame for good. Past the
   limit the pipeline is compiled without the archive. */
#define OPEN_WAIT_S 2
static id<MTLRenderPipelineState> build_pipeline(MtlPipes *p, NSData *rec, id<MTLFunction> vf, id<MTLFunction> pf,
                                                 int *from_archive, NSString **err)
{
    pthread_mutex_lock(&p->mu);
    if (!p->opened && !p->open_late) {
        struct timespec until;
        clock_gettime(CLOCK_REALTIME, &until);
        until.tv_sec += OPEN_WAIT_S;
        while (!p->opened && pthread_cond_timedwait(&p->cv, &p->mu, &until) == 0) {
        }
        if (!p->opened) {
            /* Once: later builds do not wait for it again. */
            p->open_late = 1;
            mtl_log("pipelines: the archive is not open after %d s; compiling without it", OPEN_WAIT_S);
        }
    }
    id<MTLBinaryArchive> a = p->opened ? p->archive : nil;
    int fail = p->fail;
    pthread_mutex_unlock(&p->mu);
    if (fail) {
        *from_archive = 0;
        if (err) *err = @"failure forced by the test hook";
        return nil;
    }
    *from_archive = 0;
    MTLRenderPipelineDescriptor *pd = p->describe((const uint8_t *)rec.bytes + REC_HEAD);
    pd.vertexFunction = vf;
    pd.fragmentFunction = pf;
    NSError *e = nil;
    if (a) {
        pd.binaryArchives = @[ a ];
        id<MTLRenderPipelineState> pso = [p->dev newRenderPipelineStateWithDescriptor:pd
                                                                              options:MTLPipelineOptionFailOnBinaryArchiveMiss
                                                                           reflection:nil
                                                                                error:&e];
        if (pso) {
            *from_archive = 1;
            pthread_mutex_lock(&p->mu);
            if (p->cache_dir) [p->built addObject:pd];
            pthread_mutex_unlock(&p->mu);
            return pso;
        }
        e = nil;
    }
    id<MTLRenderPipelineState> pso = [p->dev newRenderPipelineStateWithDescriptor:pd error:&e];
    if (!pso) {
        if (err) *err = e.localizedDescription;
        return nil;
    }
    pthread_mutex_lock(&p->mu);
    if (p->cache_dir) [p->built addObject:pd];
    p->cold_built++;
    pthread_mutex_unlock(&p->mu);
    return pso;
}

static void settle_pipeline(MtlPipes *p, NSData *rec, MtlPipeSlot *slot, id pso, int from_archive, int by_warm)
{
    pthread_mutex_lock(&p->mu);
    if (pso) {
        slot->obj = pso;
        slot->state = SLOT_READY;
        p->st[from_archive ? MTL_PIPES_STAT_ARCHIVE : MTL_PIPES_STAT_COLD]++;
        p->st[by_warm ? MTL_PIPES_STAT_BY_WARM : MTL_PIPES_STAT_BY_REPLAY]++;
    } else if (by_warm) {
        /* The replay thread builds it itself if a draw needs it, and logs
           why it fails if it does. */
        [p->psos removeObjectForKey:rec];
    } else {
        slot->state = SLOT_FAILED;
        p->st[MTL_PIPES_STAT_FAILED]++;
    }
    pthread_cond_broadcast(&p->cv);
    pthread_mutex_unlock(&p->mu);
}

/* The warm queue's view of a function: built from the cached MSL, since
   no shader handle exists yet. nil if the cache lacks it. */
static id<MTLFunction> warm_function(MtlPipes *p, NSString *key)
{
    pthread_mutex_lock(&p->mu);
    MtlPipeSlot *slot;
    for (;;) {
        slot = p->fns[key];
        if (!slot) break;
        if (slot->state != SLOT_BUSY) {
            id fn = slot->state == SLOT_READY ? slot->obj : nil;
            pthread_mutex_unlock(&p->mu);
            return fn;
        }
        pthread_cond_wait(&p->cv, &p->mu);
    }
    slot = [MtlPipeSlot new];
    slot->state = SLOT_BUSY;
    p->fns[key] = slot;
    pthread_mutex_unlock(&p->mu);

    NSString *msl = nil;
    int ff = [key isEqualToString:@MTL_PIPES_FF_VS] ? 1 : [key isEqualToString:@MTL_PIPES_FF_PS] ? 2 : 0;
    if (ff) {
        char err[256];
        const char *s = mtl_sc_ff_msl(ff - 1, err, sizeof err);
        msl = s ? @(s) : nil;
    } else {
        char *s = mtl_sc_cached_msl(p->cache_dir.UTF8String, key.UTF8String);
        msl = s ? @(s) : nil;
        free(s);
    }
    id<MTLFunction> fn = msl ? mtl_pipes_compile(p->dev, msl, NULL) : nil;
    pthread_mutex_lock(&p->mu);
    if (fn) {
        slot->obj = fn;
        slot->state = SLOT_READY;
        p->st[MTL_PIPES_STAT_FN_WARM]++;
    } else {
        [p->fns removeObjectForKey:key];
    }
    pthread_cond_broadcast(&p->cv);
    pthread_mutex_unlock(&p->mu);
    return fn;
}

static void warm_one(MtlPipes *p, NSData *rec)
{
    if (atomic_load(&p->cancel)) return;
    pthread_mutex_lock(&p->mu);
    if (p->psos[rec]) {
        pthread_mutex_unlock(&p->mu);
        return;
    }
    MtlPipeSlot *slot = [MtlPipeSlot new];
    slot->state = SLOT_BUSY;
    p->psos[rec] = slot;
    pthread_mutex_unlock(&p->mu);
    id<MTLFunction> vf = warm_function(p, record_stage(rec, 0));
    id<MTLFunction> pf = vf ? warm_function(p, record_stage(rec, 1)) : nil;
    int from_archive = 0;
    id<MTLRenderPipelineState> pso = vf && pf ? build_pipeline(p, rec, vf, pf, &from_archive, NULL) : nil;
    settle_pipeline(p, rec, slot, pso, from_archive, 1);
}

static void warm_all(MtlPipes *p)
{
    double t0 = now_ms();
    open_cache(p);
    for (NSData *rec in p->recorded) {
        if (atomic_load(&p->cancel)) break;
        dispatch_semaphore_wait(p->width, DISPATCH_TIME_FOREVER);
        dispatch_group_async(p->warm_jobs, p->workers, ^{
            @autoreleasepool {
                warm_one(p, rec);
            }
            dispatch_semaphore_signal(p->width);
        });
    }
    dispatch_group_wait(p->warm_jobs, DISPATCH_TIME_FOREVER);
    double ms = now_ms() - t0;
    pthread_mutex_lock(&p->mu);
    p->st[MTL_PIPES_STAT_WARM_DONE] = 1;
    p->st[MTL_PIPES_STAT_WARM_US] = (uint64_t)(ms * 1000.0);
    uint64_t made = p->st[MTL_PIPES_STAT_BY_WARM], fns = p->st[MTL_PIPES_STAT_FN_WARM];
    pthread_mutex_unlock(&p->mu);
    if (p->recorded.count)
        mtl_log("pipelines: warm queue done: %llu of %lu recorded pipelines and %llu functions in %.0f ms%s",
                (unsigned long long)made, (unsigned long)p->recorded.count, (unsigned long long)fns, ms,
                atomic_load(&p->cancel) ? " (stopped early)" : "");
}

/* Interface */

mtl_pipes *mtl_pipes_open(id<MTLDevice> dev, int archive, const char *cache_dir, const char *version, size_t desc_bytes,
                          mtl_pipes_describe describe)
{
    MtlPipes *p = [MtlPipes new];
    pthread_mutex_init(&p->mu, NULL);
    pthread_cond_init(&p->cv, NULL);
    pthread_mutex_init(&p->finish_mu, NULL);
    p->dev = dev;
    p->desc_bytes = desc_bytes;
    p->describe = describe;
    p->fns = [NSMutableDictionary new];
    p->psos = [NSMutableDictionary new];
    p->recorded = @[];
    p->used = [NSMutableOrderedSet new];
    p->built = [NSMutableArray new];
    const char *skip = getenv("RT_METAL_PSO_SKIP");
    p->skip = skip && atoi(skip) > 0;
    /* RT_METAL_PSO_IDENT stands in for another build, GPU or OS: the
       headless test uses it to check that a mismatched archive is
       thrown away. */
    const char *extra = getenv("RT_METAL_PSO_IDENT");
    p->ident = [NSString stringWithFormat:@"%s %zu, chain %s, gpu %@, os %@%s%s", version, desc_bytes,
                                          mtl_sc_chain_tag(), dev.name,
                                          [NSProcessInfo processInfo].operatingSystemVersionString,
                                          extra && *extra ? ", " : "", extra && *extra ? extra : ""];
    p->all = dispatch_group_create();
    p->warm_jobs = dispatch_group_create();
    p->width = dispatch_semaphore_create(WARM_WIDTH);
    /* User initiated, not utility: the replay thread may be waiting on
       one of these, and a condition variable passes no priority on. */
    dispatch_queue_attr_t serial =
        dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INITIATED, 0);
    dispatch_queue_attr_t conc =
        dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_CONCURRENT, QOS_CLASS_USER_INITIATED, 0);
    p->driver = dispatch_queue_create("metal_d3d9.pipelines.warm", serial);
    p->workers = dispatch_queue_create("metal_d3d9.pipelines.build", conc);
    /* Without an archive nothing about a pipeline outlives the run: the
       keys file is only ever valid beside an archive, so there is nothing
       to warm from. The MSL cache does not depend on this. */
    if (!archive && cache_dir && *cache_dir)
        mtl_log("pipelines: no binary archive on this GPU; pipelines are compiled on first use each run");
    if (archive && cache_dir && *cache_dir) {
        p->cache_dir = @(cache_dir);
        p->dir = [p->cache_dir stringByAppendingPathComponent:@(mtl_sc_chain_tag())];
        dispatch_group_async(p->all, p->driver, ^{
            @autoreleasepool {
                warm_all(p);
            }
        });
    } else {
        p->opened = 1;
        p->st[MTL_PIPES_STAT_WARM_DONE] = 1;
    }
    if (p->skip) mtl_log("pipelines: RT_METAL_PSO_SKIP=1: a draw whose pipeline is not ready is skipped for that frame");
    return (mtl_pipes *)CFBridgingRetain(p);
}

void mtl_pipes_finish(mtl_pipes *pp)
{
    if (!pp) return;
    MtlPipes *p = P(pp);
    pthread_mutex_lock(&p->finish_mu);
    /* The warm queue stops for good: past this point the run is ending,
       and a recorded pipeline it did not reach keeps its record. */
    atomic_store(&p->cancel, 1);
    dispatch_group_wait(p->all, DISPATCH_TIME_FOREVER);
    @autoreleasepool {
        if (p->cache_dir && p->archive) save_cache(p);
    }
    if (!p->reported) {
        p->reported = 1;
        uint64_t s[MTL_PIPES_STAT_N];
        mtl_pipes_stats(pp, s);
        mtl_log("pipelines: %llu used; %llu from the archive, %llu compiled cold, %llu failed; warm queue made %llu (of "
                "%llu recorded) and %llu functions in %.0f ms; replay thread made %llu and %llu functions; replay thread "
                "waited %llu times, %.1f ms in all, longest %.1f ms; %llu draws skipped",
                (unsigned long long)s[MTL_PIPES_STAT_USED], (unsigned long long)s[MTL_PIPES_STAT_ARCHIVE],
                (unsigned long long)s[MTL_PIPES_STAT_COLD], (unsigned long long)s[MTL_PIPES_STAT_FAILED],
                (unsigned long long)s[MTL_PIPES_STAT_BY_WARM], (unsigned long long)s[MTL_PIPES_STAT_RECORDED],
                (unsigned long long)s[MTL_PIPES_STAT_FN_WARM], s[MTL_PIPES_STAT_WARM_US] / 1000.0,
                (unsigned long long)s[MTL_PIPES_STAT_BY_REPLAY], (unsigned long long)s[MTL_PIPES_STAT_FN_REPLAY],
                (unsigned long long)s[MTL_PIPES_STAT_WAITS], s[MTL_PIPES_STAT_WAIT_US] / 1000.0,
                s[MTL_PIPES_STAT_WAIT_MAX_US] / 1000.0, (unsigned long long)s[MTL_PIPES_STAT_SKIPPED]);
    }
    pthread_mutex_unlock(&p->finish_mu);
}

void mtl_pipes_close(mtl_pipes *pp)
{
    if (!pp) return;
    mtl_pipes_finish(pp);
    (void)CFBridgingRelease(pp);
}

mtl_pipes_status mtl_pipes_function(mtl_pipes *pp, const char *key, NSString * (^msl)(void), id<MTLFunction> *out,
                                    mtl_pipes_wait *w, NSString **err)
{
    MtlPipes *p = P(pp);
    NSString *k = @(key);
    double t0 = now_ms();
    int waited = 0;
    *out = nil;
    pthread_mutex_lock(&p->mu);
    MtlPipeSlot *slot;
    for (;;) {
        slot = p->fns[k];
        if (!slot) break;
        if (slot->state == SLOT_READY) {
            *out = slot->obj;
            pthread_mutex_unlock(&p->mu);
            if (waited) {
                w->waited++;
                w->ms += now_ms() - t0;
            }
            return MTL_PIPES_READY;
        }
        if (slot->state == SLOT_FAILED) {
            pthread_mutex_unlock(&p->mu);
            return MTL_PIPES_FAILED;
        }
        if (p->skip) {
            pthread_mutex_unlock(&p->mu);
            return MTL_PIPES_PENDING;
        }
        waited = 1;
        pthread_cond_wait(&p->cv, &p->mu);
    }
    slot = [MtlPipeSlot new];
    slot->state = SLOT_BUSY;
    p->fns[k] = slot;
    pthread_mutex_unlock(&p->mu);

    NSString *src = msl();
    if (src && p->skip) {
        dispatch_group_async(p->all, p->workers, ^{
            @autoreleasepool {
                NSError *e = nil;
                id<MTLFunction> fn = mtl_pipes_compile(p->dev, src, &e);
                if (!fn) mtl_log("shader %s did not compile: %s", k.UTF8String, e.localizedDescription.UTF8String);
                pthread_mutex_lock(&p->mu);
                slot->obj = fn;
                slot->state = fn ? SLOT_READY : SLOT_FAILED;
                if (fn) p->st[MTL_PIPES_STAT_FN_WARM]++;
                pthread_cond_broadcast(&p->cv);
                pthread_mutex_unlock(&p->mu);
            }
        });
        w->ms += now_ms() - t0;
        return MTL_PIPES_PENDING;
    }
    NSError *e = nil;
    id<MTLFunction> fn = src ? mtl_pipes_compile(p->dev, src, &e) : nil;
    pthread_mutex_lock(&p->mu);
    slot->obj = fn;
    slot->state = fn ? SLOT_READY : SLOT_FAILED;
    if (fn) p->st[MTL_PIPES_STAT_FN_REPLAY]++;
    pthread_cond_broadcast(&p->cv);
    pthread_mutex_unlock(&p->mu);
    w->built++;
    w->ms += now_ms() - t0;
    if (!fn) {
        if (err && e) *err = e.localizedDescription;
        return MTL_PIPES_FAILED;
    }
    *out = fn;
    return MTL_PIPES_READY;
}

mtl_pipes_status mtl_pipes_pipeline(mtl_pipes *pp, const char *vs_key, const char *ps_key, const void *desc,
                                    id<MTLFunction> vf, id<MTLFunction> pf, id<MTLRenderPipelineState> *out,
                                    mtl_pipes_wait *w, NSString **err)
{
    MtlPipes *p = P(pp);
    NSData *rec = make_record(p, vs_key, ps_key, desc);
    double t0 = now_ms();
    int waited = 0;
    *out = nil;
    pthread_mutex_lock(&p->mu);
    if (![p->used containsObject:rec]) {
        [p->used addObject:rec];
        p->st[MTL_PIPES_STAT_USED] = p->used.count;
    }
    MtlPipeSlot *slot;
    for (;;) {
        slot = p->psos[rec];
        if (!slot) break;
        if (slot->state == SLOT_READY) {
            *out = slot->obj;
            pthread_mutex_unlock(&p->mu);
            if (waited) {
                w->waited++;
                w->pso_how = "warm queue";
                w->ms += now_ms() - t0;
            }
            return MTL_PIPES_READY;
        }
        if (slot->state == SLOT_FAILED) {
            pthread_mutex_unlock(&p->mu);
            return MTL_PIPES_FAILED;
        }
        if (p->skip) {
            pthread_mutex_unlock(&p->mu);
            return MTL_PIPES_PENDING;
        }
        waited = 1;
        pthread_cond_wait(&p->cv, &p->mu);
    }
    slot = [MtlPipeSlot new];
    slot->state = SLOT_BUSY;
    p->psos[rec] = slot;
    pthread_mutex_unlock(&p->mu);

    if (p->skip) {
        NSString *names = [NSString stringWithFormat:@"vs %s ps %s", vs_key, ps_key]; /* the keys are borrowed */
        dispatch_group_async(p->all, p->workers, ^{
            @autoreleasepool {
                int from_archive = 0;
                NSString *why = nil;
                id<MTLRenderPipelineState> pso = build_pipeline(p, rec, vf, pf, &from_archive, &why);
                if (!pso) mtl_log("pipeline for %s failed: %s", names.UTF8String, why.UTF8String);
                settle_pipeline(p, rec, slot, pso, from_archive, 0);
            }
        });
        return MTL_PIPES_PENDING;
    }
    int from_archive = 0;
    id<MTLRenderPipelineState> pso = build_pipeline(p, rec, vf, pf, &from_archive, err);
    settle_pipeline(p, rec, slot, pso, from_archive, 0);
    w->built++;
    w->pso_how = !pso ? "failed" : from_archive ? "archive" : "cold";
    w->ms += now_ms() - t0;
    *out = pso;
    return pso ? MTL_PIPES_READY : MTL_PIPES_FAILED;
}

void mtl_pipes_note_wait(mtl_pipes *pp, const mtl_pipes_wait *w)
{
    MtlPipes *p = P(pp);
    uint64_t us = (uint64_t)(w->ms * 1000.0);
    pthread_mutex_lock(&p->mu);
    p->st[MTL_PIPES_STAT_WAITS]++;
    p->st[MTL_PIPES_STAT_WAIT_US] += us;
    if (us > p->st[MTL_PIPES_STAT_WAIT_MAX_US]) p->st[MTL_PIPES_STAT_WAIT_MAX_US] = us;
    pthread_mutex_unlock(&p->mu);
}

void mtl_pipes_note_skipped(mtl_pipes *pp)
{
    MtlPipes *p = P(pp);
    pthread_mutex_lock(&p->mu);
    p->st[MTL_PIPES_STAT_SKIPPED]++;
    pthread_mutex_unlock(&p->mu);
}

void mtl_pipes_stats(mtl_pipes *pp, uint64_t out[MTL_PIPES_STAT_N])
{
    MtlPipes *p = P(pp);
    pthread_mutex_lock(&p->mu);
    memcpy(out, p->st, sizeof p->st);
    pthread_mutex_unlock(&p->mu);
}

void mtl_pipes_set_fail(mtl_pipes *pp, int on)
{
    MtlPipes *p = P(pp);
    pthread_mutex_lock(&p->mu);
    p->fail = on;
    pthread_mutex_unlock(&p->mu);
}

void mtl_pipes_wait_idle(mtl_pipes *pp) { dispatch_group_wait(P(pp)->all, DISPATCH_TIME_FOREVER); }
