/*
 * Stage functions and render pipelines that outlive a run (pipelines.m).
 *
 * A pipeline is recorded by what it is made of, not by the process-local
 * uids the draw path's own dictionary keys on: the two stage keys (the
 * shader's bytecode hash, or MTL_PIPES_FF_VS / MTL_PIPES_FF_PS for
 * DXVK's ubershaders) and the draw path's pipeline description, an
 * opaque block of a fixed size that the describe callback turns into a
 * descriptor without functions. That record is the persistent key.
 *
 * Under the cache directory, beside the MSL it is built from, the
 * subdirectory named by the chain tag holds pipelines.bin, an
 * MTLBinaryArchive of every pipeline made so far, and pipelines.keys,
 * the records in first-use order with an identity line (format, the
 * draw path's own version and record size, chain tag, GPU, OS) and the
 * archive's size and SHA-256. Anything that does not match is deleted
 * and rebuilt rather than trusted: Metal itself refuses a damaged
 * header but loads an archive with damaged contents and reports hits
 * from it, so the hash is the only guard against that.
 *
 * When a device is created, a warm queue opens the archive and builds
 * every recorded pipeline off the game and render threads, up to three
 * at a time. The replay side asks for functions and pipelines through
 * the calls below: ready ones come back at once; one the warm queue is
 * building is waited for; one nobody has started is built by the
 * caller. With RT_METAL_PSO_SKIP=1 the caller never waits: what is not
 * ready is handed to the warm queue and the draw is skipped for that
 * frame. Waiting is the default because a skipped draw is a wrong
 * frame.
 *
 * Objective-C only; everything here is safe from any thread.
 */
#ifndef MTL_PIPELINES_H
#define MTL_PIPELINES_H

#import <Metal/Metal.h>

#include <stddef.h>
#include <stdint.h>

#define MTL_PIPES_STAGE_KEY 32 /* bytes per stage key in a record, NUL padded */
#define MTL_PIPES_FF_VS "ff-vs"
#define MTL_PIPES_FF_PS "ff-ps"

typedef struct mtl_pipes mtl_pipes;

typedef enum {
    MTL_PIPES_READY,
    MTL_PIPES_PENDING, /* skip mode: being built elsewhere, not ready this frame */
    MTL_PIPES_FAILED,
} mtl_pipes_status;

/* What one draw's lookups cost the calling thread, for its log line. */
typedef struct {
    double ms;           /* spent building or waiting */
    uint32_t built;      /* functions or pipelines built by the caller */
    uint32_t waited;     /* waited for the warm queue */
    const char *pso_how; /* "archive", "cold", "warm queue", or NULL when the pipeline was ready */
} mtl_pipes_wait;

/* The describe callback: a descriptor with vertex layout, attachments
   and blending from a recorded description, functions left unset. */
typedef MTLRenderPipelineDescriptor *(*mtl_pipes_describe)(const void *desc);

/* Opens the cache under cache_dir (nil: nothing persists and nothing is
   warmed) and starts warming. archive 0 (the GPU has none, mtl_gpu_caps)
   means the same as no cache_dir, and no MTLBinaryArchive call is ever
   made. version names the draw path's description format and must
   change whenever the description or describe does. */
mtl_pipes *mtl_pipes_open(id<MTLDevice> dev, int archive, const char *cache_dir, const char *version, size_t desc_bytes,
                          mtl_pipes_describe describe);

/* Stops warming, writes both files when anything changed since they
   were read or last written (a new archive of every pipeline made this
   run, and the records), and logs the counters the first time. The
   cache stays usable, so this is the process exit hook's half of a
   close: safe to call again, and before mtl_pipes_close. */
void mtl_pipes_finish(mtl_pipes *p);

/* mtl_pipes_finish, then frees. */
void mtl_pipes_close(mtl_pipes *p);

/* A stage function by key. msl is called on this thread only when the
   function must be built here or handed off, and returns the source or
   nil (it logs its own failure). On FAILED, *err says why, if known. */
mtl_pipes_status mtl_pipes_function(mtl_pipes *p, const char *key, NSString * (^msl)(void), id<MTLFunction> *out,
                                    mtl_pipes_wait *w, NSString **err);

/* A pipeline by stage keys and description, with the functions those
   keys name. */
mtl_pipes_status mtl_pipes_pipeline(mtl_pipes *p, const char *vs_key, const char *ps_key, const void *desc,
                                    id<MTLFunction> vf, id<MTLFunction> pf, id<MTLRenderPipelineState> *out,
                                    mtl_pipes_wait *w, NSString **err);

/* A draw whose lookups were not all ready at once: counted, with the
   time, as replay thread time spent waiting on creation. */
void mtl_pipes_note_wait(mtl_pipes *p, const mtl_pipes_wait *w);

/* A draw skipped because something it needed was not ready. */
void mtl_pipes_note_skipped(mtl_pipes *p);

/* The counters, in the order metal_d3d9_pipeline_stats reports them. */
enum {
    MTL_PIPES_STAT_OPEN,          /* 0 no cache, 1 new, 2 loaded, 3 corrupt and rebuilt, 4 mismatched and rebuilt */
    MTL_PIPES_STAT_RECORDED,      /* records read from pipelines.keys */
    MTL_PIPES_STAT_ARCHIVE,       /* pipelines made from the archive, by any thread */
    MTL_PIPES_STAT_COLD,          /* pipelines compiled with no archive hit, by any thread */
    MTL_PIPES_STAT_BY_WARM,       /* pipelines made by the warm queue */
    MTL_PIPES_STAT_BY_REPLAY,     /* pipelines made by the replay thread itself */
    MTL_PIPES_STAT_WAITS,         /* replay lookups that were not ready at once */
    MTL_PIPES_STAT_WAIT_US,       /* replay thread time in them */
    MTL_PIPES_STAT_WAIT_MAX_US,
    MTL_PIPES_STAT_SKIPPED,       /* draws skipped (RT_METAL_PSO_SKIP=1) */
    MTL_PIPES_STAT_FN_WARM,       /* functions compiled by the warm queue */
    MTL_PIPES_STAT_FN_REPLAY,     /* functions compiled by the replay thread */
    MTL_PIPES_STAT_WARM_DONE,     /* 1 once the warm queue has nothing left */
    MTL_PIPES_STAT_USED,          /* distinct pipelines this run asked for */
    MTL_PIPES_STAT_WARM_US,       /* the warm queue's wall time, open included */
    MTL_PIPES_STAT_FAILED,        /* pipelines that failed to build */
    MTL_PIPES_STAT_N
};
void mtl_pipes_stats(mtl_pipes *p, uint64_t out[MTL_PIPES_STAT_N]);

/* Test: from now on every pipeline build fails, as a GPU that rejects a
   pipeline would make it, so the draw path's handling can be checked. */
void mtl_pipes_set_fail(mtl_pipes *p, int on);

/* Blocks until the warm queue, and any work skip mode handed it, is done. */
void mtl_pipes_wait_idle(mtl_pipes *p);

/* MSL to a function, with the options every stage is compiled with. */
id<MTLFunction> mtl_pipes_compile(id<MTLDevice> dev, NSString *msl, NSError **e);

#endif
