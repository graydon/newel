use packed_simd::{FromCast, Simd};
use rayon::prelude::*;
use std::ops::*;
use std::mem::size_of;

use crate::zeroone::*;
use crate::traits::*;
use crate::consts::*;

#[derive(Clone, Debug, PartialEq)]
pub enum BoolBinOpCode {
    Lt,
    Le,
    Eq,
    Ne,
    Ge,
    Gt,
}

#[derive(Clone, Debug, PartialEq)]
pub enum BoolUnOpCode {
    IsNaN,
    IsInf,
    IsFin,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ValBinOpCode {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Min,
    Max,
    Pow,
    BitAnd,
    BitOr,
    BitXor,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ValUnOpCode {
    Neg,
    BitNot,
    Abs,
    Ln,
    Exp,
    Sqrt,
    Sin,
    Cos,
}

impl_unop_skel!((u8, u8) (u8, bool)
                (u16, u16) (u16, bool)
                (u32, u32) (u32, bool)
                (u64, u64) (u64, bool)
                (u128, u128) (u128, bool)
                (i8, i8) (i8, bool)
                (i16, i16) (i16, bool)
                (i32, i32) (i32, bool)
                (i64, i64) (i64, bool)
                (i128, i128) (i128, bool)
                (f32, f32) (f32, bool)
                (f64, f64) (f64, bool));

impl_binop_skel!((u8, u8) (u8, bool)
                 (u16, u16) (u16, bool)
                 (u32, u32) (u32, bool)
                 (u64, u64) (u64, bool)
                 (u128, u128) (u128, bool)
                 (i8, i8) (i8, bool)
                 (i16, i16) (i16, bool)
                 (i32, i32) (i32, bool)
                 (i64, i64) (i64, bool)
                 (i128, i128) (i128, bool)
                 (f32, f32) (f32, bool)
                 (f64, f64) (f64, bool));

// Signed-only unops.
impl_unop!(NegOp, neg, i8 i16 i32 i64 i128 f32 f64);

// Unimplemented on bools and unsigned types.
impl_unop_unsupported!(NegOp, bool u8 u16 u32 u64 u128);


// Floating-point-only unops
impl_unop!(SinOp, sin, f32 f64);
impl_unop!(CosOp, cos, f32 f64);
impl_unop!(LnOp, ln, f32 f64);
impl_unop!(ExpOp, exp, f32 f64);
impl_unop!(SqrtOp, sqrt, f32 f64);
impl_unop!(AbsOp, abs, f32 f64);

// The FP-only ops are unsupported for bools or integers.
impl_unop_unsupported!(SinOp, bool u8 u16 u32 u64 u128 i8 i16 i32 i64 i128);
impl_unop_unsupported!(CosOp, bool u8 u16 u32 u64 u128 i8 i16 i32 i64 i128);
impl_unop_unsupported!(LnOp, bool u8 u16 u32 u64 u128 i8 i16 i32 i64 i128);
impl_unop_unsupported!(ExpOp, bool u8 u16 u32 u64 u128 i8 i16 i32 i64 i128);
impl_unop_unsupported!(SqrtOp, bool u8 u16 u32 u64 u128 i8 i16 i32 i64 i128);
impl_unop_unsupported!(AbsOp, bool u8 u16 u32 u64 u128 i8 i16 i32 i64 i128);

// Unary predicates.
impl_unop_pred!(IsNaNOp, is_nan, f32 f64);
impl_unop_pred!(IsInfOp, is_infinite, f32 f64);
impl_unop_pred!(IsFinOp, is_finite, f32 f64);

// The FP-only predicates are unsupported for integers or bools.
impl_unop_pred_unsupported!(IsNaNOp, bool u8 u16 u32 u64 u128 i8 i16 i32 i64 i128);
impl_unop_pred_unsupported!(IsInfOp, bool u8 u16 u32 u64 u128 i8 i16 i32 i64 i128);
impl_unop_pred_unsupported!(IsFinOp, bool u8 u16 u32 u64 u128 i8 i16 i32 i64 i128);

// Integer-only unops
impl_unop!(NotOp, not, u8 u16 u32 u64 u128 i8 i16 i32 i64 i128);

// Integer-only unops not supported for bool or FP.
impl_unop_unsupported!(NotOp, bool f32 f64);

// General arithmetic binops.
impl_binop!(AddOp, add, u8 u16 u32 u64 u128 i8 i16 i32 i64 i128 f32 f64);
impl_binop!(SubOp, sub, u8 u16 u32 u64 u128 i8 i16 i32 i64 i128 f32 f64);

impl_binop!(MulOp, mul, u8 u16 u32 u64 u128 i8 i16 i32 i64 i128 f32 f64);
impl_binop!(DivOp, div, u8 u16 u32 u64 u128 i8 i16 i32 i64 i128 f32 f64);
impl_binop!(RemOp, rem, u8 u16 u32 u64 u128 i8 i16 i32 i64 i128 f32 f64);

impl_binop!(MinOp, min, u8 u16 u32 u64 u128 i8 i16 i32 i64 i128 f32 f64);
impl_binop!(MaxOp, max, u8 u16 u32 u64 u128 i8 i16 i32 i64 i128 f32 f64);

// General arithmetic binops aren't supported by bools.
impl_binop_unsupported!(AddOp, bool);
impl_binop_unsupported!(SubOp, bool);
impl_binop_unsupported!(MulOp, bool);
impl_binop_unsupported!(DivOp, bool);
impl_binop_unsupported!(RemOp, bool);
impl_binop_unsupported!(MinOp, bool);
impl_binop_unsupported!(MaxOp, bool);


// Floating-point-only binops.
impl_binop!(PowOp, powf, f32 f64);

// FP-only binop is not supported by bool or integer types.
impl_binop_unsupported!(PowOp, bool u8 u16 u32 u64 u128 i8 i16 i32 i64 i128);

// Integer-only binops.
impl_binop!(BitAndOp, bitand, u8 u16 u32 u64 u128 i8 i16 i32 i64 i128);
impl_binop!(BitOrOp, bitor, u8 u16 u32 u64 u128 i8 i16 i32 i64 i128);
impl_binop!(BitXorOp, bitxor, u8 u16 u32 u64 u128 i8 i16 i32 i64 i128);

// Integer-only ops are not supported by bool or FP.
impl_binop_unsupported!(BitAndOp, bool f32 f64);
impl_binop_unsupported!(BitOrOp, bool f32 f64);
impl_binop_unsupported!(BitXorOp, bool f32 f64);

// Binary predicates.
impl_binop_pred!(LtOp, lt, u8 u16 u32 u64 u128 i8 i16 i32 i64 i128 f32 f64);
impl_binop_pred!(LeOp, le, u8 u16 u32 u64 u128 i8 i16 i32 i64 i128 f32 f64);
impl_binop_pred!(EqOp, eq, u8 u16 u32 u64 u128 i8 i16 i32 i64 i128 f32 f64);
impl_binop_pred!(NeOp, ne, u8 u16 u32 u64 u128 i8 i16 i32 i64 i128 f32 f64);
impl_binop_pred!(GeOp, ge, u8 u16 u32 u64 u128 i8 i16 i32 i64 i128 f32 f64);
impl_binop_pred!(GtOp, gt, u8 u16 u32 u64 u128 i8 i16 i32 i64 i128 f32 f64);

impl_binop_pred_unsupported!(LtOp, bool);
impl_binop_pred_unsupported!(LeOp, bool);
impl_binop_pred_unsupported!(EqOp, bool);
impl_binop_pred_unsupported!(NeOp, bool);
impl_binop_pred_unsupported!(GeOp, bool);
impl_binop_pred_unsupported!(GtOp, bool);

// The type-conversion operator.
pub struct ConvOp<SRC, DST> {
    _x: std::marker::PhantomData<(SRC, DST)>,
}

impl_convop!(u8, u16 u32 u64 u128 i8 i16 i32 i64 i128 f32 f64);
impl_convop!(u16, u8 u32 u64 u128 i8 i16 i32 i64 i128 f32 f64);
impl_convop!(u32, u8 u16 u64 u128 i8 i16 i32 i64 i128 f32 f64);
impl_convop!(u64, u8 u16 u32 u128 i8 i16 i32 i64 i128 f32 f64);
impl_convop!(u128, u8 u16 u32 u64 i8 i16 i32 i64 i128 f32 f64);

impl_convop!(i8, u8 u16 u32 u64 u128 i16 i32 i64 i128 f32 f64);
impl_convop!(i16, u8 u16 u32 u64 u128 i8 i32 i64 i128 f32 f64);
impl_convop!(i32, u8 u16 u32 u64 u128 i8 i16 i64 i128 f32 f64);
impl_convop!(i64, u8 u16 u32 u64 u128 i8 i16 i32 i128 f32 f64);
impl_convop!(i128, u8 u16 u32 u64 u128 i8 i16 i32 i64 f32 f64);

impl_convop!(f32, u8 u16 u32 u64 u128 i8 i16 i32 i64 i128 f64);
impl_convop!(f64, u8 u16 u32 u64 u128 i8 i16 i32 i64 i128 f32);

impl_noop_convop!(bool u8 u16 u32 u64 u128 i8 i16 i32 i64 i128 f32 f64);

impl_bool_convop!(u8 u16 u32 u64 u128 i8 i16 i32 i64 i128 f32 f64);
