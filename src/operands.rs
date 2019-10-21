use crate::scalarty::*;

/// Operands are the primary types of arguments passed to newel's evaluator and
/// returned from it after operations complete. They are either single-element
/// constants or homogeneous slices.
#[derive(Clone, Debug, PartialEq)]
pub enum Operand<'a> {
    OperandSlice(Slice<'a>),
    OperandConst(Const)
}

#[derive(Clone, Debug, PartialEq)]
pub enum Const {
    ConstBool(bool),
    ConstU8(u8),
    ConstU16(u16),
    ConstU32(u32),
    ConstU64(u64),
    ConstU128(u128),
    ConstI8(i8),
    ConstI16(i16),
    ConstI32(i32),
    ConstI64(i64),
    ConstI128(i128),
    ConstF32(f32),
    ConstF64(f64),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Slice<'a> {
    SliceBool(&'a [bool]),
    SliceU8(&'a [u8]),
    SliceU16(&'a [u16]),
    SliceU32(&'a [u32]),
    SliceU64(&'a [u64]),
    SliceU128(&'a [u128]),
    SliceI8(&'a [i8]),
    SliceI16(&'a [i16]),
    SliceI32(&'a [i32]),
    SliceI64(&'a [i64]),
    SliceI128(&'a [i128]),
    SliceF32(&'a [f32]),
    SliceF64(&'a [f64]),
}


impl Const {
    pub fn get_scalar_ty(&self) -> ScalarTy {
        use Const::*;
        use ScalarTy::*;
        match self {
            ConstBool(_) => TBool,
            ConstU8(_) => TU8,
            ConstU16(_) => TU16,
            ConstU32(_) => TU32,
            ConstU64(_) => TU64,
            ConstU128(_) => TU128,
            ConstI8(_) => TI8,
            ConstI16(_) => TI16,
            ConstI32(_) => TI32,
            ConstI64(_) => TI64,
            ConstI128(_) => TI128,
            ConstF32(_) => TF32,
            ConstF64(_) => TF64,
       }
    }
}

impl<'a> Slice<'a> {
    pub fn get_scalar_ty(&self) -> ScalarTy {
        use Slice::*;
        use ScalarTy::*;
        match self {
            SliceBool(_) => TBool,
            SliceU8(_) => TU8,
            SliceU16(_) => TU16,
            SliceU32(_) => TU32,
            SliceU64(_) => TU64,
            SliceU128(_) => TU128,
            SliceI8(_) => TI8,
            SliceI16(_) => TI16,
            SliceI32(_) => TI32,
            SliceI64(_) => TI64,
            SliceI128(_) => TI128,
            SliceF32(_) => TF32,
            SliceF64(_) => TF64,
        }
    }
}

impl<'a> Operand<'a> {
    pub fn get_scalar_ty(&self) -> ScalarTy {
        use Operand::*;
        match self {
            OperandConst(c) => c.get_scalar_ty(),
            OperandSlice(c) => c.get_scalar_ty()
        }
    }
}

impl_operand_from!([bool] ConstBool SliceBool
                   [u8] ConstU8 SliceU8
                   [u16] ConstU16 SliceU16
                   [u32] ConstU32 SliceU32
                   [u64] ConstU64 SliceU64
                   [u128] ConstU128 SliceU128
                   [i8] ConstI8 SliceI8
                   [i16] ConstI16 SliceI16
                   [i32] ConstI32 SliceI32
                   [i64] ConstI64 SliceI64
                   [i128] ConstI128 SliceI128
                   [f32] ConstF32 SliceF32
                   [f64] ConstF64 SliceF64
);

