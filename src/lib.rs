// Copyright 2019-2020 Graydon Hoare <graydon@pobox.com>
// Licensed under the MIT and Apache-2.0 licenses.

/*!
 * This crate provides a set of bulk homogeneous-slice-typed operations. It's to
 * be used as a building block for collection-oriented evaluators that don't
 * JIT, but still want good performance from a fixed repertoire of
 * type-specialized inner loops. It provides three notable features:
 *
 *   1. Isolates in one compilation unit the code amplification factor of the
 *      cartesian product of all primitive operations across all primitive
 *      types, compiled into (where possible) efficient SIMD loops. They only
 *      get compiled once, here.
 *
 *   2. Abstracts away the question of runtime selection of the most efficient
 *      implementation (eg. rayon chunking & SIMD, possibly GPU in the future).
 *
 *   3. Provides "outer" column-vs-column or vs-constant dynamic
 *      double-dispatches that are expected to be amortized over the inner
 *      loops, as well as the dynamic dispatch from ops to functions. Nothing in
 *      this crate should wind up inlined much into its users, despite going
 *      very fast once inside!
 *
 * This is the kind of building block used in eg. APL evaluators or
 * column-oriented databases, where the outer layer is an _interpreter_ and it
 * maintains good-but-not-maximal performance by having all the _inner_ layers
 * go fast: the homogeneous-slice-typed operators here. For data intensive
 * programs it's a simple strategy that seems to pay off well, since the outer
 * interpreter loop is only returned to relatively infrequently, and the code
 * footprint of these inner loops is compiled once into the interpreter, so
 * constant across all user programs and relatively cache-friendly.
 */


#[macro_use]
mod macros;

mod traits;
mod zeroone;
mod consts;
mod operands;
mod scalarty;
mod ops;
mod eval;
mod tests;

// These are the public API. Intentionally narrow and dynamically-typed.
pub use consts::{CHUNKBYTES,VECBYTES};
pub use scalarty::ScalarTy;
pub use operands::{Const,Slice,Operand};
pub use ops::{BoolBinOpCode,BoolUnOpCode,ValBinOpCode,ValUnOpCode};
pub use eval::{EvalError,EvalCtx};

// TODO:
//   1. DONE: Switch from slices to discriminated union of constant-or-slice.
//   2. DONE: Pass tmp buffer.
//   3. DONE: Figure out how to chain ConvSliceFn to an operator.
//   4. DONE: Figure out how to select a promotion type for a pair of input types.
//   5. DONE: Reduce the per-op amplification to exclude the rayon part; only
//            switch op at the per-chunk level, factor out "skeletons".
//   6. DONE: Implement convops for bools in terms of cmpops against consts.
//   7. DONE: Implement const+slice, slice+const, const+const variants.
//   8. DONE: Figure out API pattern for missing / nonsense ops (& other errors).
//   9. DONE: Modularize a bit.
//  10. DONE: Minimize transmute badness, use/enhance/reimplement safe_transmute.
//  11. DONE: Revisit buffer size calculations, expected and required sizes.
//  12. DONE: Audit access control.
//  13. DONE: Rename things to have less-silly names.
//  14. LATER: Add non-SIMD fallback macros for ops not in packed_simd.
//  15. LATER: Add decimal128.
//  16. LATER: Add packed small-string types / ops.
//  17. LATER: Add features to make a small or full-sized version.
//  18. LATER: Figure out how best to trap ubiquitous faults like SIGFPE.
//  19. DONE: Mop up egregious warnings / clippy-isms.
//  20. DONE: At least a handful of tests.
//  21. DONE: At least a handful of benchmarks.

