// Copyright 2019-2020 Graydon Hoare <graydon@pobox.com>
// Licensed under the MIT and Apache-2.0 licenses.

pub trait ScalarT: Sized {}
impl ScalarT for bool {}
impl ScalarT for u8 {}
impl ScalarT for u16 {}
impl ScalarT for u32 {}
impl ScalarT for u64 {}
impl ScalarT for u128 {}
impl ScalarT for i8 {}
impl ScalarT for i16 {}
impl ScalarT for i32 {}
impl ScalarT for i64 {}
impl ScalarT for i128 {}
impl ScalarT for f32 {}
impl ScalarT for f64 {}

pub enum OpError {
    Unsupported,
}

pub trait UnOp<SRC: ScalarT, DST: ScalarT> {
    fn apply_slice<'src, 'dst>(src: &'src [SRC],
                               dst: &'dst mut [DST])
                               -> Result<&'dst [DST], OpError>
    where
        'src: 'dst;

    fn apply_const(src: SRC) -> Result<DST, OpError>;
}

pub trait BinOp<SRC: ScalarT, DST: ScalarT> {
    fn apply_slice_slice<'src, 'dst>(lhs: &'src [SRC],
                                     rhs: &'src [SRC],
                                     dst: &'dst mut [DST])
                                     -> Result<&'dst [DST], OpError>
    where
        'src: 'dst;

    fn apply_slice_const<'src, 'dst>(lhs: &'src [SRC],
                                     rhs: SRC,
                                     dst: &'dst mut [DST])
                                     -> Result<&'dst [DST], OpError>
    where
        'src: 'dst;

    fn apply_const_slice<'src, 'dst>(lhs: SRC,
                                     rhs: &'src [SRC],
                                     dst: &'dst mut [DST])
                                     -> Result<&'dst [DST], OpError>
    where
        'src: 'dst;

    fn apply_const_const(lhs: SRC, rhs: SRC) -> Result<DST, OpError>;
}
