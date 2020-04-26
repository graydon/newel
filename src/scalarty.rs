// Copyright 2019-2020 Graydon Hoare <graydon@pobox.com>
// Licensed under the MIT and Apache-2.0 licenses.

/// All `Operand`s in newel have an underlying `ScalarTy` which is the type of
/// every element in a homogeneous `Slice` operand, or the type of the sole
/// element of a `Const` operand. The `ScalarTy` is
// dynamically inspected in order to select the operation body and target type
/// for any evaluation step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarTy {
    TBool,
    TU8,
    TU16,
    TU32,
    TU64,
    TU128,
    TI8,
    TI16,
    TI32,
    TI64,
    TI128,
    TF32,
    TF64,
}

impl ScalarTy {

    /// Returns the ScalarTy that's the join of `self` and `other`: should be
    /// used to decide the type to promote `self` and `other` to when used as
    /// the two types of a binary operator.
    pub fn join(self, other: Self) -> Self {
        use ScalarTy::*;
        match (self, other) {
            // unsigned \/ unsigned
            (TU8, TU16) => TU16,
            (TU8, TU32) => TU32,
            (TU8, TU64) => TU64,
            (TU8, TU128) => TU128,
            (TU16, TU32) => TU32,
            (TU16, TU64) => TU64,
            (TU16, TU128) => TU128,
            (TU32, TU64) => TU64,
            (TU32, TU128) => TU128,
            (TU64, TU128) => TU128,

            // signed \/ signed
            (TI8, TI16) => TI16,
            (TI8, TI32) => TI32,
            (TI8, TI64) => TI64,
            (TI8, TI128) => TI128,
            (TI16, TI32) => TI32,
            (TI16, TI64) => TI64,
            (TI16, TI128) => TI128,
            (TI32, TI64) => TI64,
            (TI32, TI128) => TI128,
            (TI64, TI128) => TI128,

            // unsigned \/ signed
            (TU8, TI8) => TI16,
            (TU8, TI16) => TI16,
            (TU8, TI32) => TI32,
            (TU8, TI64) => TI64,
            (TU8, TI128) => TI128,
            (TU16, TI16) => TI32,
            (TU16, TI32) => TI32,
            (TU16, TI64) => TI64,
            (TU16, TI128) => TI128,
            (TU32, TI32) => TI64,
            (TU32, TI64) => TI64,
            (TU32, TI128) => TI128,
            (TU64, TI64) => TI128,
            (TU64, TI128) => TI128,
            (TU128, TI128) => TF64,

            // float
            (TF32, TU8) => TF32,
            (TF32, TU16) => TF32,
            (TF32, TU32) => TF32,
            (TF32, TU64) => TF64,
            (TF32, TU128) => TF64,
            (TF32, TI8) => TF32,
            (TF32, TI16) => TF32,
            (TF32, TI32) => TF32,
            (TF32, TI64) => TF64,
            (TF32, TI128) => TF64,

            (TF64, _) => TF64,

            (a, b) if a == b => a,
            (x, y) => y.join(x),
        }
    }
}
