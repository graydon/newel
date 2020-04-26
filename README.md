# Newel

This crate provides a set of bulk homogeneous-slice-typed operations. It's to
be used as a building block for collection-oriented evaluators that don't
JIT, but still want good performance from a fixed repertoire of
type-specialized inner loops. It provides three notable features:

  1. Isolates in one compilation unit the code amplification factor of the
     cartesian product of all primitive operations across all primitive
     types, compiled into (where possible) efficient SIMD loops. They only
     get compiled once, here.

  2. Abstracts away the question of runtime selection of the most efficient
     implementation (eg. rayon chunking & SIMD, possibly GPU in the future).

  3. Provides "outer" column-vs-column or vs-constant dynamic
     double-dispatches that are expected to be amortized over the inner
     loops, as well as the dynamic dispatch from ops to functions. Nothing in
     this crate should wind up inlined much into its users, despite going
     very fast once inside!

This is the kind of building block used in eg. APL evaluators or
column-oriented databases, where the outer layer is an _interpreter_ and it
maintains good-but-not-maximal performance by having all the _inner_ layers
go fast: the homogeneous-slice-typed operators here. For data intensive
programs it's a simple strategy that seems to pay off well, since the outer
interpreter loop is only returned to relatively infrequently, and the code
footprint of these inner loops is compiled once into the interpreter, so
constant across all user programs and relatively cache-friendly.

## Reference

T. Kersten, V. Leis, A. Kemper, T. Neumann, A. Pavlo, P. Boncz.
Everything You Always Wanted to Know About Compiled and Vectorized
Queries But Were Afraid to Ask. PVLDB, 11 (13): 2209 - 2222, 2018.

DOI: https://doi.org/10.14778/3275366.3275370

http://www.vldb.org/pvldb/vol11/p2209-kersten.pdf

## Performance

Benchmarks operating on pairs of 16MB columns.

SIMD speedup over scalar loops:

~~~
$ RAYON_NUM_THREADS=1 cargo bench
test add_u32_newel  ... bench:   1,056,316 ns/iter (+/- 455,605) = 3970 MB/s
test add_u32_scalar ... bench:   2,099,241 ns/iter (+/- 37,008) = 1998 MB/s
test add_u64_newel  ... bench:   2,265,895 ns/iter (+/- 300,628) = 3702 MB/s
test add_u64_scalar ... bench:   2,202,050 ns/iter (+/- 84,681) = 3809 MB/s
test add_u8_newel   ... bench:     346,318 ns/iter (+/- 58,334) = 3027 MB/s
test add_u8_scalar  ... bench:   2,089,541 ns/iter (+/- 52,697) = 501 MB/s
test lt_u16_newel   ... bench:     392,897 ns/iter (+/- 74,161) = 5337 MB/s
test lt_u16_scalar  ... bench:   1,800,218 ns/iter (+/- 85,009) = 1164 MB/s
test lt_u32_newel   ... bench:     636,441 ns/iter (+/- 122,298) = 6590 MB/s
test lt_u32_scalar  ... bench:   2,101,893 ns/iter (+/- 63,897) = 1995 MB/s
test lt_u8_newel    ... bench:     335,296 ns/iter (+/- 45,917) = 3127 MB/s
test lt_u8_scalar   ... bench:   2,105,158 ns/iter (+/- 57,694) = 498 MB/s
~~~

Plus thread-level parallel speedup:

~~~
$ RAYON_NUM_THREADS=40 cargo bench
test add_u32_newel  ... bench:     163,534 ns/iter (+/- 31,830) = 25647 MB/s
test add_u32_scalar ... bench:   2,112,953 ns/iter (+/- 87,481) = 1985 MB/s
test add_u64_newel  ... bench:     283,384 ns/iter (+/- 58,405) = 29601 MB/s
test add_u64_scalar ... bench:   2,254,990 ns/iter (+/- 104,615) = 3720 MB/s
test add_u8_newel   ... bench:      78,888 ns/iter (+/- 19,092) = 13291 MB/s
test add_u8_scalar  ... bench:   2,099,180 ns/iter (+/- 70,613) = 499 MB/s
test lt_u16_newel   ... bench:      99,430 ns/iter (+/- 30,703) = 21091 MB/s
test lt_u16_scalar  ... bench:   1,821,691 ns/iter (+/- 58,804) = 1151 MB/s
test lt_u32_newel   ... bench:     129,451 ns/iter (+/- 39,019) = 32400 MB/s
test lt_u32_scalar  ... bench:   2,102,977 ns/iter (+/- 27,062) = 1994 MB/s
test lt_u8_newel    ... bench:      73,589 ns/iter (+/- 20,625) = 14249 MB/s
test lt_u8_scalar   ... bench:   2,114,156 ns/iter (+/- 82,354) = 495 MB/s
~~~

## Name

Wikipedia:

> A newel, also called a central pole or support column, is the central supporting pillar of a staircase.

