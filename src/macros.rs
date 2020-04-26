// Copyright 2019-2020 Graydon Hoare <graydon@pobox.com>
// Licensed under the MIT and Apache-2.0 licenses.

////////////////////////////////////////////////////////////////////////////////
// convert::From support for operands
////////////////////////////////////////////////////////////////////////////////

macro_rules! impl_operand_from {
    ($([$T:ty] $const_ctor:ident $slice_ctor:ident)*) => {
        $(
            impl std::convert::From<$T> for Const {
                fn from(s: $T) -> Self {
                    Const::$const_ctor(s)
                }
            }
            impl<'a> std::convert::From<&'a [$T]> for Slice<'a> {
                fn from(s: &'a [$T]) -> Self {
                    Slice::$slice_ctor(s)
                }
            }
            impl<'a> std::convert::From<$T> for Operand<'a> {
                fn from(s: $T) -> Self {
                    Operand::OperandConst(Const::$const_ctor(s))
                }
            }
            impl<'a> std::convert::From<&'a [$T]> for Operand<'a> {
                fn from(s: &'a [$T]) -> Self {
                    Operand::OperandSlice(Slice::$slice_ctor(s))
                }
            }
            impl<'a> std::convert::From<&'a Vec<$T>> for Operand<'a> {
                fn from(s: &'a Vec<$T>) -> Self {
                    Operand::OperandSlice(Slice::$slice_ctor(s.as_slice()))
                }
            }
        )*
    }
}


////////////////////////////////////////////////////////////////////////////////
// Operator skeletons
////////////////////////////////////////////////////////////////////////////////
//
// To reduce the amount of code amplification, we make a common outer
// "skeletons" of rayon CHUNKSZ-walking code for each set of types (and for both
// unop and binop cases), and then dispatch once *dynamically* for each chunk to
// an inner single-CHUNKSZ operator function.

macro_rules! impl_unop_skel {
    ($(($SRC:ty, $DST:ty))*) => {
        pub struct UnOpSkel<SRC, DST> {
            _x: std::marker::PhantomData<(SRC,DST)>,
        }
        $(
            impl UnOpSkel<$SRC, $DST>
            {
                #[inline(never)]
                fn skel<'src, 'dst>(src: &'src [$SRC],
                                    dst: &'dst mut[$DST],
                                    f: &(dyn Sync + Fn(&[$SRC], &mut [$DST])))
                {
                    const CHUNKSZ : usize = chunksz_min::<$SRC,$DST>();
                    let len = src.len();
                    assert_eq!(len, dst.len());
                    assert_eq!((len & !(CHUNKSZ-1)), len);
                    src.par_chunks(CHUNKSZ)
                        .zip(dst.par_chunks_mut(CHUNKSZ))
                        .for_each(|(srcchunk, dstchunk)|
                                  f(srcchunk, dstchunk));
                }
            }
        )*
    }
}

macro_rules! impl_binop_skel {
    ($(($SRC:ty , $DST:ty))*) => {
        pub struct BinOpSkel<SRC,DST> {
            _x: std::marker::PhantomData<(SRC,DST)>,
        }
        $(
            impl BinOpSkel<$SRC,$DST>
            {
                #[inline(never)]
                fn skel<'src, 'dst>(lhs: &'src [$SRC],
                                    rhs: &'src [$SRC],
                                    dst: &'dst mut[$DST],
                                    f: &(dyn Sync + Fn(&[$SRC], &[$SRC], &mut [$DST])))
                where 'src: 'dst,
                {
                    const CHUNKSZ : usize = chunksz_min::<$SRC,$DST>();
                    let len = rhs.len();
                    assert_eq!(len, lhs.len());
                    assert_eq!(len, dst.len());
                    assert_eq!((len & !(CHUNKSZ-1)), len);
                    lhs.par_chunks(CHUNKSZ)
                        .zip(rhs.par_chunks(CHUNKSZ))
                        .zip(dst.par_chunks_mut(CHUNKSZ))
                        .for_each(|((lhschunk,rhschunk), dstchunk)|
                                  f(lhschunk, rhschunk, dstchunk));
                }
            }
        )*
    }
}

////////////////////////////////////////////////////////////////////////////////
// Unary T->T operators
////////////////////////////////////////////////////////////////////////////////

macro_rules! impl_unop {
    ($struct_id:ident, $op:ident, $($T:ty)*) => {
        pub struct $struct_id<SRC, DST> {
            _x: std::marker::PhantomData<(SRC,DST)>,
        }

        $(
            impl UnOp<$T,$T> for $struct_id<$T,$T>
            {
                #[inline(never)]
                fn apply_const(src: $T) -> Result<$T, OpError>
                {
                    Ok(src.$op())
                }

                #[inline(never)]
                fn apply_slice<'src, 'dst>(src: &'src [$T],
                                           dst: &'dst mut[$T])
                                           -> Result<&'dst [$T], OpError>
                where 'src: 'dst
                {
                    const STEPSZ : usize = VECBYTES / size_of::<$T>();
                    <UnOpSkel<$T,$T>>::skel(
                        src, dst,
                        &|srcchunk, dstchunk| {
                            for (src, dst) in
                                srcchunk.chunks_exact(STEPSZ)
                                .zip(dstchunk.chunks_exact_mut(STEPSZ))
                            {
                                let sv = <Simd<[$T; STEPSZ]>>::from_slice_unaligned(src);
                                let dv = sv.$op();
                                dv.write_to_slice_unaligned(dst)
                            }
                        });
                    Ok(dst)
                }
            }
        )*
    }
}

////////////////////////////////////////////////////////////////////////////////
// Unary T->bool ("predicate") operators
////////////////////////////////////////////////////////////////////////////////

macro_rules! impl_unop_pred {
    ($struct_id:ident, $op:ident, $($T:ty)*) => {
        pub struct $struct_id<SRC,DST> {
            _x: std::marker::PhantomData<(SRC,DST)>,
        }

        $(
            impl UnOp<$T,bool> for $struct_id<$T,bool>
            {
                #[inline(never)]
                fn apply_const(src: $T) -> Result<bool, OpError>
                {
                    Ok(src.$op())
                }

                #[inline(never)]
                fn apply_slice<'src, 'dst>(src: &'src [$T],
                                           dst: &'dst mut[bool])
                                           -> Result<&'dst [bool], OpError>
                where 'src: 'dst
                {
                    const STEPSZ : usize = stepsz_min::<$T,bool>();
                    const TRUES : Simd<[u8; STEPSZ]> = <Simd<[u8; STEPSZ]>>::splat(1);
                    const FALSES : Simd<[u8; STEPSZ]> = <Simd<[u8; STEPSZ]>>::splat(0);
                    <UnOpSkel<$T,bool>>::skel(
                        src, dst,
                        &|srcchunk, dstchunk| {
                            for (src, dst) in
                                srcchunk.chunks_exact(STEPSZ)
                                .zip(dstchunk.chunks_exact_mut(STEPSZ))
                            {
                                let sv = <Simd<[$T; STEPSZ]>>::from_slice_unaligned(src);
                                let mv = sv.$op();
                                let bv = mv.select(TRUES, FALSES);
                                unsafe {
                                    bv.write_to_slice_unaligned(::std::mem::transmute::<&mut[bool],&mut[u8]>(dst));
                                }
                            }
                        });
                    Ok(dst)
                }
            }
        )*
    }
}

////////////////////////////////////////////////////////////////////////////////
// Binary (T,T)->T operators
////////////////////////////////////////////////////////////////////////////////

macro_rules! impl_binop {
    ($struct_id:ident, $op:ident, $($T:ty)*) => {
        pub struct $struct_id<SRC, DST> {
            _x: std::marker::PhantomData<(SRC,DST)>,
        }
        $(
            impl BinOp<$T,$T> for $struct_id<$T,$T>
            {
                #[inline(never)]
                fn apply_slice_slice<'src, 'dst>(lhs: &'src [$T],
                                                 rhs: &'src [$T],
                                                 dst: &'dst mut[$T])
                                                 -> Result<&'dst [$T], OpError>
                where
                'src: 'dst,
                {
                    const STEPSZ : usize = VECBYTES / size_of::<$T>();
                    <BinOpSkel<$T,$T>>::skel(
                        lhs, rhs, dst,
                        &|lhschunk, rhschunk, dstchunk| {
                            for ((lhs, rhs), dst) in
                                lhschunk.chunks_exact(STEPSZ)
                                .zip(rhschunk.chunks_exact(STEPSZ))
                                .zip(dstchunk.chunks_exact_mut(STEPSZ))
                            {
                                let lv = <Simd<[$T; STEPSZ]>>::from_slice_unaligned(lhs);
                                let rv = <Simd<[$T; STEPSZ]>>::from_slice_unaligned(rhs);
                                let dv = lv.$op(rv);
                                dv.write_to_slice_unaligned(dst);
                            }
                        });
                    Ok(dst)
                }

                #[inline(never)]
                fn apply_slice_const<'src, 'dst>(lhs: &'src [$T],
                                                 rhs: $T,
                                                 dst: &'dst mut[$T])
                                                 -> Result<&'dst [$T], OpError>
                where
                'src: 'dst,
                {
                    const STEPSZ : usize = VECBYTES / size_of::<$T>();
                    let rv = <Simd<[$T; STEPSZ]>>::splat(rhs);
                    <UnOpSkel<$T,$T>>::skel(
                        lhs, dst,
                        &|lhschunk, dstchunk| {
                            for (lhs, dst) in
                                lhschunk.chunks_exact(STEPSZ)
                                .zip(dstchunk.chunks_exact_mut(STEPSZ))
                            {
                                let lv = <Simd<[$T; STEPSZ]>>::from_slice_unaligned(lhs);
                                let dv = lv.$op(rv);
                                dv.write_to_slice_unaligned(dst);
                            }
                        });
                    Ok(dst)
                }

                #[inline(never)]
                fn apply_const_slice<'src, 'dst>(lhs: $T,
                                                 rhs: &'src [$T],
                                                 dst: &'dst mut[$T])
                                                 -> Result<&'dst [$T], OpError>
                where
                'src: 'dst,
                {
                    const STEPSZ : usize = VECBYTES / size_of::<$T>();
                    let lv = <Simd<[$T; STEPSZ]>>::splat(lhs);
                    <UnOpSkel<$T,$T>>::skel(
                        rhs, dst,
                        &|rhschunk, dstchunk| {
                            for (rhs, dst) in
                                rhschunk.chunks_exact(STEPSZ)
                                .zip(dstchunk.chunks_exact_mut(STEPSZ))
                            {
                                let rv = <Simd<[$T; STEPSZ]>>::from_slice_unaligned(rhs);
                                let dv = lv.$op(rv);
                                dv.write_to_slice_unaligned(dst);
                            }
                        });
                    Ok(dst)
                }

                #[inline(never)]
                fn apply_const_const(lhs: $T, rhs: $T) -> Result<$T, OpError>
                {
                    Ok(lhs.$op(rhs))
                }
            }
        )*
    }
}

////////////////////////////////////////////////////////////////////////////////
// Unary (T)->U unsupported operators
////////////////////////////////////////////////////////////////////////////////
macro_rules! impl_unop_unsupported_full {
    ($struct_id:ident, $(($T:ty, $U:ty))*) => {
        $(
            impl UnOp<$T,$U> for $struct_id<$T,$U>
            {
                #[inline(never)]
                fn apply_const(_src: $T) -> Result<$U, OpError>
                {
                    Err(OpError::Unsupported)
                }
                #[inline(never)]
                fn apply_slice<'src, 'dst>(_src: &'src [$T],
                                           _dst: &'dst mut[$U])
                                           -> Result<&'dst [$U], OpError>
                where 'src: 'dst
                {
                    Err(OpError::Unsupported)
                }
            }
        )*
    }
}

macro_rules! impl_unop_unsupported {
    ($struct_id:ident, $($T:ty)*) => {
        impl_unop_unsupported_full!($struct_id, $(($T, $T))*);
    }
}

macro_rules! impl_unop_pred_unsupported {
    ($struct_id:ident, $($T:ty)*) => {
        impl_unop_unsupported_full!($struct_id, $(($T, bool))*);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Binary (T,T)->U unsupported operators
////////////////////////////////////////////////////////////////////////////////

macro_rules! impl_binop_unsupported_full {
    ($struct_id:ident, $(($T:ty, $U:ty))*) => {
        $(
            impl BinOp<$T,$U>
                for
                $struct_id<$T,$U>
            {
                #[inline(never)]
                fn apply_slice_slice<'src, 'dst>(_lhs: &'src [$T],
                                                 _rhs: &'src [$T],
                                                 _dst: &'dst mut[$U])
                                                 -> Result<&'dst [$U], OpError>
                where
                    'src: 'dst,
                {
                    Err(OpError::Unsupported)
                }

                #[inline(never)]
                fn apply_const_slice<'src, 'dst>(_lhs: $T,
                                                 _rhs: &'src [$T],
                                                 _dst: &'dst mut[$U])
                                                 -> Result<&'dst [$U], OpError>
                where
                    'src: 'dst,
                {
                    Err(OpError::Unsupported)
                }

                #[inline(never)]
                fn apply_slice_const<'src, 'dst>(_lhs: &'src [$T],
                                                 _rhs: $T,
                                                 _dst: &'dst mut[$U])
                                                 -> Result<&'dst [$U], OpError>
                where
                    'src: 'dst,
                {
                    Err(OpError::Unsupported)
                }

                #[inline(never)]
                fn apply_const_const(_lhs: $T, _rhs: $T) -> Result<$U, OpError>
                {
                    Err(OpError::Unsupported)
                }
            }
        )*
    }
}

macro_rules! impl_binop_unsupported {
    ($struct_id:ident, $($T:ty)*) => {
        impl_binop_unsupported_full!($struct_id, $(($T, $T))*);
    }
}

macro_rules! impl_binop_pred_unsupported {
    ($struct_id:ident, $($T:ty)*) => {
        impl_binop_unsupported_full!($struct_id, $(($T, bool))*);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Binary (T,T)->bool ("comparison") operators
////////////////////////////////////////////////////////////////////////////////

macro_rules! impl_binop_pred {
    ($struct_id:ident, $op:ident, $($T:ty)*) => {

        pub struct $struct_id<SRC,DST>
        {
            _x: std::marker::PhantomData<(SRC,DST)>,
        }

        $(
            impl BinOp<$T,bool>
                for
                $struct_id<$T,bool>
            {
                #[inline(never)]
                fn apply_slice_slice<'src, 'dst>(lhs: &'src [$T],
                                                 rhs: &'src [$T],
                                                 dst: &'dst mut[bool])
                                                 -> Result<&'dst [bool], OpError>
                where
                    'src: 'dst,
                {
                    const STEPSZ : usize = stepsz_min::<$T,bool>();
                    const TRUES : Simd<[u8; STEPSZ]> = <Simd<[u8; STEPSZ]>>::splat(1);
                    const FALSES : Simd<[u8; STEPSZ]> = <Simd<[u8; STEPSZ]>>::splat(0);
                    <BinOpSkel<$T,bool>>::skel(
                        lhs, rhs, dst,
                        &|lhschunk, rhschunk, dstchunk| {
                            for ((lhs, rhs), dst) in
                                lhschunk.chunks_exact(STEPSZ)
                                .zip(rhschunk.chunks_exact(STEPSZ))
                                .zip(dstchunk.chunks_exact_mut(STEPSZ)) {
                                    let lv = <Simd<[$T; STEPSZ]>>::from_slice_unaligned(lhs);
                                    let rv = <Simd<[$T; STEPSZ]>>::from_slice_unaligned(rhs);
                                    let mv = lv.$op(rv);
                                    let bv = mv.select(TRUES, FALSES);
                                    unsafe {
                                        bv.write_to_slice_unaligned(::std::mem::transmute::<&mut[bool],&mut[u8]>(dst));
                                    }
                                }
                        });
                    Ok(dst)
                }

                #[inline(never)]
                fn apply_const_slice<'src, 'dst>(lhs: $T,
                                                 rhs: &'src [$T],
                                                 dst: &'dst mut[bool])
                                                 -> Result<&'dst [bool], OpError>
                where
                    'src: 'dst,
                {
                    const STEPSZ : usize = stepsz_min::<$T,bool>();
                    const TRUES : Simd<[u8; STEPSZ]> = <Simd<[u8; STEPSZ]>>::splat(1);
                    const FALSES : Simd<[u8; STEPSZ]> = <Simd<[u8; STEPSZ]>>::splat(0);
                    let lv = <Simd<[$T; STEPSZ]>>::splat(lhs);
                    <UnOpSkel<$T,bool>>::skel(
                        rhs, dst,
                        &|rhschunk, dstchunk| {
                            for (rhs, dst) in
                                rhschunk.chunks_exact(STEPSZ)
                                .zip(dstchunk.chunks_exact_mut(STEPSZ)) {
                                    let rv = <Simd<[$T; STEPSZ]>>::from_slice_unaligned(rhs);
                                    let mv = lv.$op(rv);
                                    let bv = mv.select(TRUES, FALSES);
                                    unsafe {
                                        bv.write_to_slice_unaligned(::std::mem::transmute::<&mut[bool],&mut[u8]>(dst));
                                    }
                                }
                        });
                    Ok(dst)
                }

                #[inline(never)]
                fn apply_slice_const<'src, 'dst>(lhs: &'src [$T],
                                                 rhs: $T,
                                                 dst: &'dst mut[bool])
                                                 -> Result<&'dst [bool], OpError>
                where
                    'src: 'dst,
                {
                    const STEPSZ : usize = stepsz_min::<$T,bool>();
                    const TRUES : Simd<[u8; STEPSZ]> = <Simd<[u8; STEPSZ]>>::splat(1);
                    const FALSES : Simd<[u8; STEPSZ]> = <Simd<[u8; STEPSZ]>>::splat(0);
                    let rv = <Simd<[$T; STEPSZ]>>::splat(rhs);
                    <UnOpSkel<$T,bool>>::skel(
                        lhs, dst,
                        &|lhschunk, dstchunk| {
                            for (lhs, dst) in
                                lhschunk.chunks_exact(STEPSZ)
                                .zip(dstchunk.chunks_exact_mut(STEPSZ)) {
                                    let lv = <Simd<[$T; STEPSZ]>>::from_slice_unaligned(lhs);
                                    let mv = lv.$op(rv);
                                    let bv = mv.select(TRUES, FALSES);
                                    unsafe {
                                        bv.write_to_slice_unaligned(::std::mem::transmute::<&mut[bool],&mut[u8]>(dst));
                                    }
                                }
                        });
                    Ok(dst)
                }

                #[inline(never)]
                fn apply_const_const(lhs: $T, rhs: $T) -> Result<bool, OpError>
                {
                    Ok(lhs.$op(&rhs))
                }
            }
        )*
    }
}

////////////////////////////////////////////////////////////////////////////////
// Conversion operators
////////////////////////////////////////////////////////////////////////////////

macro_rules! impl_noop_convop {
    ($($T:ty)*) => {
        $(
            impl UnOp<$T,$T> for ConvOp<$T, $T>
            {
                #[inline(never)]
                fn apply_const(src: $T) -> Result<$T, OpError>
                {
                    Ok(src)
                }
                #[inline(never)]
                fn apply_slice<'src, 'dst>(src: &'src [$T],
                                           _dst: &'dst mut[$T])
                                           -> Result<&'dst [$T], OpError>
                where
                    'src: 'dst
                {
                    Ok(src)
                }
            }
        )*
    }
}

macro_rules! impl_convop {
    ($SRC:ty, $($DST:ty)*) => {
        $(
            impl UnOp<$SRC, $DST> for ConvOp<$SRC, $DST>
            {
                #[inline(never)]
                fn apply_const(src: $SRC) -> Result<$DST, OpError>
                {
                    const STEPSZ : usize = stepsz_min::<$SRC,$DST>();
                    let sv = <Simd<[$SRC; STEPSZ]>>::splat(src);
                    let dv = <Simd<[$DST; STEPSZ]>>::from_cast(sv);
                    Ok(dv.extract(0))
                }

                #[inline(never)]
                fn apply_slice<'src, 'dst>(src: &'src [$SRC],
                                           dst: &'dst mut[$DST])
                                           -> Result<&'dst [$DST], OpError>
                where
                    'src: 'dst
                {
                    const STEPSZ : usize = stepsz_min::<$SRC,$DST>();
                    const CHUNKSZ : usize = chunksz_min::<$SRC,$DST>();
                    let len = src.len();
                    assert_eq!(len, dst.len());
                    assert_eq!((len & !(CHUNKSZ-1)), len);
                    src.par_chunks(CHUNKSZ)
                        .zip(dst.par_chunks_mut(CHUNKSZ))
                        .for_each(|(srcchunk, dstchunk)|
                                  {
                                      for (src, dst) in
                                          srcchunk.chunks_exact(STEPSZ)
                                          .zip(dstchunk.chunks_exact_mut(STEPSZ))
                                      {
                                          let sv = <Simd<[$SRC; STEPSZ]>>::from_slice_unaligned(src);
                                          let dv = <Simd<[$DST; STEPSZ]>>::from_cast(sv);
                                          dv.write_to_slice_unaligned(dst);
                                      }
                                  });
                    Ok(dst)
                }
            }
        )*
    }
}

macro_rules! impl_bool_convop {
    ($($T:ty)*) => {
        $(
            impl UnOp<$T, bool> for ConvOp<$T, bool>
            {
                #[inline(never)]
                fn apply_const(src: $T) -> Result<bool, OpError>
                {
                    Ok(src.ne(&<$T>::ZERO))
                }
                #[inline(never)]
                fn apply_slice<'src, 'dst>(src: &'src [$T],
                                           dst: &'dst mut[bool])
                                           -> Result<&'dst [bool], OpError>
                where
                    'src: 'dst
                {
                    const STEPSZ : usize = stepsz_min::<$T,bool>();
                    const CHUNKSZ : usize = chunksz_min::<$T,bool>();
                    const TRUES : Simd<[u8; STEPSZ]> = <Simd<[u8; STEPSZ]>>::splat(1);
                    const FALSES : Simd<[u8; STEPSZ]> = <Simd<[u8; STEPSZ]>>::splat(0);
                    const ZEROES : Simd<[$T; STEPSZ]> = <Simd<[$T; STEPSZ]>>::splat(<$T>::ZERO);

                    let len = src.len();
                    assert_eq!(len, dst.len());
                    assert_eq!((len & !(CHUNKSZ-1)), len);
                    src.par_chunks(CHUNKSZ)
                        .zip(dst.par_chunks_mut(CHUNKSZ))
                        .for_each(|(srcchunk, dstchunk)|
                                  {
                                      for (src, dst) in
                                          srcchunk.chunks_exact(STEPSZ)
                                          .zip(dstchunk.chunks_exact_mut(STEPSZ))
                                      {
                                          let sv = <Simd<[$T; STEPSZ]>>::from_slice_unaligned(src);
                                          let mv = sv.ne(ZEROES);
                                          let bv = mv.select(TRUES, FALSES);
                                          unsafe {
                                              bv.write_to_slice_unaligned(::std::mem::transmute::<&mut[bool],&mut[u8]>(dst));
                                          }
                                      }
                                  });
                    Ok(dst)
                }
            }

            impl UnOp<bool, $T> for ConvOp<bool, $T>
            {
                #[inline(never)]
                fn apply_const(src: bool) -> Result<$T, OpError>
                {
                    Ok(if src { <$T>::ONE } else { <$T>::ZERO })
                }
                #[inline(never)]
                fn apply_slice<'src, 'dst>(src: &'src [bool],
                                           dst: &'dst mut[$T])
                                           -> Result<&'dst [$T], OpError>
                where
                    'src: 'dst
                {
                    const STEPSZ : usize = stepsz_min::<$T,bool>();
                    const CHUNKSZ : usize = chunksz_min::<$T,bool>();
                    const ZEROES : Simd<[$T; STEPSZ]> = <Simd<[$T; STEPSZ]>>::splat(<$T>::ZERO);
                    const ONES : Simd<[$T; STEPSZ]> = <Simd<[$T; STEPSZ]>>::splat(<$T>::ONE);

                    let len = src.len();
                    assert_eq!(len, dst.len());
                    assert_eq!((len & !(CHUNKSZ-1)), len);
                    src.par_chunks(CHUNKSZ)
                        .zip(dst.par_chunks_mut(CHUNKSZ))
                        .for_each(|(srcchunk, dstchunk)|
                                  {
                                      for (src, dst) in
                                          srcchunk.chunks_exact(STEPSZ)
                                          .zip(dstchunk.chunks_exact_mut(STEPSZ))
                                      {
                                          let uv = unsafe {
                                              <Simd<[u8; STEPSZ]>>::from_slice_unaligned(
                                                  ::std::mem::transmute::<&[bool],&[u8]>(src))
                                          };
                                          let mv = <Simd<[packed_simd::m8; STEPSZ]>>::from_cast(uv);
                                          let bv = mv.select(ONES, ZEROES);
                                          bv.write_to_slice_unaligned(dst);
                                      }
                                  });
                    Ok(dst)
                }
            }
        )*
    }
}
