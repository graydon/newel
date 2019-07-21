use std::mem::size_of;

// We write all our SIMD code over 0x40-byte = 64-byte = 512-bit vectors, which
// are natively supported on some targets but usually decay to some set of
// smaller vector sizes on most (eg. on plain x64/SSE we'll wind up with what
// looks like a 4-level-unrolled loop using 128-bit XMM registers). This makes
// our code much easier to write, at the cost of losing a bit of finer-grained
// control over exactly what gets emitted. Not that .. unrolling is bad anyway?
//
// (This is debatable: LLVM's auto-vectorizer _often_ does a good job picking
// the right loop size, such that in a few cases it seems that it beats the
// explicit Simd<...> calls here; on the other hand in other cases involving
// mask vectors it mostly doesn't, and it's hard to predict. For simplicity
// and predictability sake we go with explicit Simd<...> always.)
//
// Conveniently 64 bytes is also the L1 cache line size which is unlikely to
// change especially soon (see below). It's also the number of *bits* in a
// modern machine word, and we wish to have the flexibility of working with
// either two possible representations for boolean vectors: one-bit-per-boolean
// (64 bit word) and one-byte-per-boolean (64 byte cache line). They are
// both useful in different circumstances.
pub const VECBYTES: usize = 0x40;

// L1 caches are (somewhat universally) 32k, at least so long as caches are
// 8-way associative VIPT and pages are 4096 bytes. There's a much more complex
// overview of the relationship here but it seems like an acceptable thing to
// bake into the design:
// https://stackoverflow.com/questions/46480015/vipt-cache-connection-between-tlb-cache
//
// (ARM cores have different associativity and indexing/tagging but seem also to
// like sitting on the 64-byte-line / 32k L1 design point, so *shrug*)
//
// Given that we typically are operating on two such columns at a time, we could
// try to make our rayon-parallel column chunk size 16k; but this only gives the
// per-chunk SIMD loops 256 iterations before returning to Rayon, which is not
// quite enough time to amortize its overhead. Seems the sweet spot is around
// 1024 iterations, or 64k chunks. This does not appear to antagonize the L1
// cache any worse than we did by falling out into Rayon too early. Oh well!
pub const CHUNKBYTES: usize = 0x10000;

// Hacky workaround for "no control flow ops in const at present": calculates
// the maximum of two usizes, as a const fn.
pub const fn cmax(a: usize, b: usize) -> usize {
    [a, b][(a < b) as usize]
}

// Loops over pairs of vectors with different-sized scalar types have to be done
// in a common "minimal" step size that's the fundamental vector size divided by
// the _largest_ of any argument vector-component sizes. This is not a
// theoretical limitation or anything; just that practically, packed_simd only
// implements operator variants up-to "512 bits" regardless of how it's
// decomposed: there are no virtual 1024-bit or 2048-bit vectors in it.
pub const fn stepsz_min<T, U>() -> usize
where
    T: std::marker::Sized,
    U: std::marker::Sized,
{
    VECBYTES / cmax(size_of::<T>(), size_of::<U>())
}

// Similarly loops over pairs of (Rayon) chunks with different-sized scalar
// types are done in a common "minimal" chunk size. This is for a different
// reason: we _could_ pass any chunk size we wanted to Rayon, but for the sake
// of a (hopefully) uniform cache-performance / work-granularity profile we aim
// to have all the different operand combinations use (as close as possible to)
// the same number of bytes in their Rayon chunks.
pub const fn chunksz_min<T, U>() -> usize
where
    T: std::marker::Sized,
    U: std::marker::Sized,
{
    CHUNKBYTES / cmax(size_of::<T>(), size_of::<U>())
}
