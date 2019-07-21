use crate::ops::*;
use crate::traits::*;
use crate::scalarty::*;
use crate::operands::*;

#[derive(Debug)]
pub enum EvalError {
    /// Not every operand type is compatible with every opcode; if the
    /// combination is unsupported, an `EvalError::UnsupportedOp` is returned.
    UnsupportedOp,
    /// The buffers held in EvalCtx need to be correctly sized and aligned to
    /// accommodate the input and output operands. If any buffer sizes are
    /// wrong, or one of the buffer transmutes fails for some other reason (bad
    /// alignment or inadequate size) an `EvalError::BadBuffer` is returned.
    BadBuffer
}

// Returns the count of `T` elements that fit in x, or error if
// `x` is not an even multiple of the size of `T` or is not aligned
// to the alignment of `T`.
fn check_align_and_size<T>(x: &[u8]) -> Result<usize, EvalError>
{
    use core::mem::{align_of, size_of};
    let a = x.as_ptr() as usize;
    if (a % align_of::<T>()) != 0 {
        return Err(EvalError::BadBuffer)
    }
    let n = x.len();
    if (n % size_of::<T>()) != 0 {
        return Err(EvalError::BadBuffer)
    }
    Ok(n / size_of::<T>())
}

fn transmute_buf_mut<T>(x: &mut [u8]) -> Result<&mut [T], EvalError>
where T:ScalarT
{
    let m = check_align_and_size::<T>(x)?;
    let p = x.as_mut_ptr() as *mut T;
    Ok(unsafe { core::slice::from_raw_parts_mut(p, m) })
}

// Checks that `x` and `y` have the same length, or returns an
// error.
fn check_equal_lengths<T>(x: &[T], y: &[T]) -> Result<(), EvalError>
{
    let n = x.len();
    if n != y.len() {
        return Err(EvalError::BadBuffer);
    }
    Ok(())
}

// Checks that `x` is an exact multiple of `consts::VECBYTES / sizeof(T)`
// and `consts::CHUNKBYTES / sizeof(T)`, returns error if not.
fn check_ok_length<T>(x: &[T]) -> Result<(), EvalError> {
    let n = x.len();
    let sz: usize = core::mem::size_of::<T>();
    let ch: usize = crate::consts::CHUNKBYTES / sz;
    let v: usize = crate::consts::VECBYTES / sz;
    if (n & (ch - 1)) != 0 {
        return Err(EvalError::BadBuffer);
    }
    if (n & (v - 1)) != 0 {
        return Err(EvalError::BadBuffer);
    }
    Ok(())
}

// Returns `x` sliced-down to to the length of `bound`, or error if
// `x` is less than `bound`.
fn bound_output_length<'a, T, U>(x: &'a mut [T], bound: &[U]) -> Result<&'a mut [T], EvalError>
{
    check_ok_length(bound)?;
    let n = x.len();
    let k = bound.len();
    if n < k {
        return Err(EvalError::BadBuffer);
    }
    Ok(&mut x[0..k])
}

/// Every top-level evaluation step in newel happens against an EvalCtx that
/// holds 3 mutable buffers of some multiple of CHUNKSZ bytes. These buffers
/// (and the EvalCtx itself) get _used up_ during the operation. The first two
/// buffers are for holding possible conversions of 1 or 2 inputs to the
/// operation; the third buffer stores the output.
pub struct EvalCtx<'eval> {
    pub tmp1: &'eval mut [u8],
    pub tmp2: &'eval mut [u8],
    pub out: &'eval mut [u8],
}

impl<'eval> EvalCtx<'eval> {

    /// Convert an `Operand` to a given `ScalarTy`.
    pub fn conv<'slice: 'eval>(self, s: &Operand<'slice>, ty: ScalarTy)
                           -> Result<Operand<'eval>, EvalError> {
        use Operand::*;
        let ok = match s {
            OperandSlice(s) => OperandSlice(self.conv_slice(s, ty)?),
            OperandConst(c) => OperandConst(self.conv_const(c, ty)?),
        };
        Ok(ok)
    }

    fn conv_const(self, c: &Const, ty: ScalarTy) -> Result<Const, EvalError> {
        use Const::*;
        use ScalarTy::*;
        let ok = match ty {
            TBool => ConstBool(conv_const_dynamic(c)?),
            TU8 => ConstU8(conv_const_dynamic(c)?),
            TU16 => ConstU16(conv_const_dynamic(c)?),
            TU32 => ConstU32(conv_const_dynamic(c)?),
            TU64 => ConstU64(conv_const_dynamic(c)?),
            TU128 => ConstU128(conv_const_dynamic(c)?),
            TI8 => ConstI8(conv_const_dynamic(c)?),
            TI16 => ConstI16(conv_const_dynamic(c)?),
            TI32 => ConstI32(conv_const_dynamic(c)?),
            TI64 => ConstI64(conv_const_dynamic(c)?),
            TI128 => ConstI128(conv_const_dynamic(c)?),
            TF32 => ConstF32(conv_const_dynamic(c)?),
            TF64 => ConstF64(conv_const_dynamic(c)?),
        };
        Ok(ok)
    }

    #[inline(never)]
    fn conv_slice<'slice>(self, s: &Slice<'slice>, ty: ScalarTy)
                          -> Result<Slice<'eval>, EvalError>
    where 'slice: 'eval
    {
        use transmute_buf_mut as tm;
        use Slice::*;
        use ScalarTy::*;
        let ok = match ty {
            TBool => SliceBool(conv_slice_dynamic(s, tm(self.out)?)?),
            TU8 => SliceU8(conv_slice_dynamic(s, tm(self.out)?)?),
            TU16 => SliceU16(conv_slice_dynamic(s, tm(self.out)?)?),
            TU32 => SliceU32(conv_slice_dynamic(s, tm(self.out)?)?),
            TU64 => SliceU64(conv_slice_dynamic(s, tm(self.out)?)?),
            TU128 => SliceU128(conv_slice_dynamic(s, tm(self.out)?)?),
            TI8 => SliceI8(conv_slice_dynamic(s, tm(self.out)?)?),
            TI16 => SliceI16(conv_slice_dynamic(s, tm(self.out)?)?),
            TI32 => SliceI32(conv_slice_dynamic(s, tm(self.out)?)?),
            TI64 => SliceI64(conv_slice_dynamic(s, tm(self.out)?)?),
            TI128 => SliceI128(conv_slice_dynamic(s, tm(self.out)?)?),
            TF32 => SliceF32(conv_slice_dynamic(s, tm(self.out)?)?),
            TF64 => SliceF64(conv_slice_dynamic(s, tm(self.out)?)?),
        };
        Ok(ok)
    }

    /// Perform a given `ValBinOpCode` on a pair of `Operand`s.
    #[inline(never)]
    pub fn val_binop<'slice>(self, op: ValBinOpCode,
                             lhs: &Operand<'slice>,
                             rhs: &Operand<'slice>)
                             -> Result<Operand<'eval>, EvalError>

    // NB: this is a bit counterintuitive, but the input lifetime 'slice has to
    // outlive the evaluator lifetime 'eval (or at least some putative output
    // lifetime, which we're currently just identifying with 'eval) because it's
    // possible that one or more of the conversion steps in the evaluation will
    // be a no-op and just returns its input.
    where 'slice: 'eval
    {
        use ScalarTy::*;
        let common_ty = lhs.get_scalar_ty().join(rhs.get_scalar_ty());
        match common_ty {
            TBool => self.val_binop_static::<bool>(op, lhs, rhs),
            TU8 => self.val_binop_static::<u8>(op, lhs, rhs),
            TU16 => self.val_binop_static::<u16>(op, lhs, rhs),
            TU32 => self.val_binop_static::<u32>(op, lhs, rhs),
            TU64 => self.val_binop_static::<u64>(op, lhs, rhs),
            TU128 => self.val_binop_static::<u128>(op, lhs, rhs),
            TI8 => self.val_binop_static::<i8>(op, lhs, rhs),
            TI16 => self.val_binop_static::<i16>(op, lhs, rhs),
            TI32 => self.val_binop_static::<i32>(op, lhs, rhs),
            TI64 => self.val_binop_static::<i64>(op, lhs, rhs),
            TI128 => self.val_binop_static::<i128>(op, lhs, rhs),
            TF32 => self.val_binop_static::<f32>(op, lhs, rhs),
            TF64 => self.val_binop_static::<f64>(op, lhs, rhs),
        }
    }

    /// Perform a given `BoolBinOpCode` on a pair of `Operand`s.
    #[inline(never)]
    pub fn bool_binop<'slice>(self, op: BoolBinOpCode,
                              lhs: &Operand<'slice>,
                              rhs: &Operand<'slice>)
                              -> Result<Operand<'eval>, EvalError>
    where 'slice: 'eval
    {
        use ScalarTy::*;
        let common_ty = lhs.get_scalar_ty().join(rhs.get_scalar_ty());
        match common_ty {
            TBool => self.bool_binop_static::<bool>(op, lhs, rhs),
            TU8 => self.bool_binop_static::<u8>(op, lhs, rhs),
            TU16 => self.bool_binop_static::<u16>(op, lhs, rhs),
            TU32 => self.bool_binop_static::<u32>(op, lhs, rhs),
            TU64 => self.bool_binop_static::<u64>(op, lhs, rhs),
            TU128 => self.bool_binop_static::<u128>(op, lhs, rhs),
            TI8 => self.bool_binop_static::<i8>(op, lhs, rhs),
            TI16 => self.bool_binop_static::<i16>(op, lhs, rhs),
            TI32 => self.bool_binop_static::<i32>(op, lhs, rhs),
            TI64 => self.bool_binop_static::<i64>(op, lhs, rhs),
            TI128 => self.bool_binop_static::<i128>(op, lhs, rhs),
            TF32 => self.bool_binop_static::<f32>(op, lhs, rhs),
            TF64 => self.bool_binop_static::<f64>(op, lhs, rhs),
        }
    }

    /// Perform a given `ValUnOpCode` on a given `Operand`.
    #[inline(never)]
    pub fn val_unop<'slice>(self, op: ValUnOpCode,
                            operand: &Operand<'slice>)
                            -> Result<Operand<'eval>, EvalError>
    where 'slice: 'eval
    {
        use ScalarTy::*;
        match operand.get_scalar_ty() {
            TBool => self.val_unop_static::<bool>(op, operand),
            TU8 => self.val_unop_static::<u8>(op, operand),
            TU16 => self.val_unop_static::<u16>(op, operand),
            TU32 => self.val_unop_static::<u32>(op, operand),
            TU64 => self.val_unop_static::<u64>(op, operand),
            TU128 => self.val_unop_static::<u128>(op, operand),
            TI8 => self.val_unop_static::<i8>(op, operand),
            TI16 => self.val_unop_static::<i16>(op, operand),
            TI32 => self.val_unop_static::<i32>(op, operand),
            TI64 => self.val_unop_static::<i64>(op, operand),
            TI128 => self.val_unop_static::<i128>(op, operand),
            TF32 => self.val_unop_static::<f32>(op, operand),
            TF64 => self.val_unop_static::<f64>(op, operand),
        }
    }

    /// Perform a given `BoolUnOpCode` on a given `Operand`.
    #[inline(never)]
    pub fn bool_unop<'slice>(self, op: BoolUnOpCode,
                              operand: &Operand<'slice>)
                              -> Result<Operand<'eval>, EvalError>
    where 'slice: 'eval
    {
        use ScalarTy::*;
        match operand.get_scalar_ty() {
            TBool => self.bool_unop_static::<bool>(op, operand),
            TU8 => self.bool_unop_static::<u8>(op, operand),
            TU16 => self.bool_unop_static::<u16>(op, operand),
            TU32 => self.bool_unop_static::<u32>(op, operand),
            TU64 => self.bool_unop_static::<u64>(op, operand),
            TU128 => self.bool_unop_static::<u128>(op, operand),
            TI8 => self.bool_unop_static::<i8>(op, operand),
            TI16 => self.bool_unop_static::<i16>(op, operand),
            TI32 => self.bool_unop_static::<i32>(op, operand),
            TI64 => self.bool_unop_static::<i64>(op, operand),
            TI128 => self.bool_unop_static::<i128>(op, operand),
            TF32 => self.bool_unop_static::<f32>(op, operand),
            TF64 => self.bool_unop_static::<f64>(op, operand),
        }
    }

    #[inline(never)]
    fn val_binop_static<'slice, T>(self, op: ValBinOpCode,
                                   lhs: &Operand<'slice>,
                                   rhs: &Operand<'slice>)
                                   -> Result<Operand<'eval>, EvalError>
    where
        'slice: 'eval,
        T: 'eval,
        T: ScalarT,
        Slice<'slice>: From<&'eval [T]>,
        Const: From<T>,

        AddOp<T, T>: BinOp<T, T>,
        SubOp<T, T>: BinOp<T, T>,
        MulOp<T, T>: BinOp<T, T>,
        DivOp<T, T>: BinOp<T, T>,
        RemOp<T, T>: BinOp<T, T>,
        MinOp<T, T>: BinOp<T, T>,
        MaxOp<T, T>: BinOp<T, T>,
        PowOp<T, T>: BinOp<T, T>,
        BitAndOp<T, T>: BinOp<T, T>,
        BitOrOp<T, T>: BinOp<T, T>,
        BitXorOp<T, T>: BinOp<T, T>,

        ConvOp<bool, T>: UnOp<bool, T>,
        ConvOp<u8, T>: UnOp<u8, T>,
        ConvOp<u16, T>: UnOp<u16, T>,
        ConvOp<u32, T>: UnOp<u32, T>,
        ConvOp<u64, T>: UnOp<u64, T>,
        ConvOp<u128, T>: UnOp<u128, T>,
        ConvOp<i8, T>: UnOp<i8, T>,
        ConvOp<i16, T>: UnOp<i16, T>,
        ConvOp<i32, T>: UnOp<i32, T>,
        ConvOp<i64, T>: UnOp<i64, T>,
        ConvOp<i128, T>: UnOp<i128, T>,
        ConvOp<f32, T>: UnOp<f32, T>,
        ConvOp<f64, T>: UnOp<f64, T>,
    {
        use Operand::*;
        use ValBinOpCode::*;
        use transmute_buf_mut as tm;
        match (lhs, rhs) {
            (OperandSlice(lhs), OperandSlice(rhs)) => {
                let tlhs: &mut [T] = tm(self.tmp1)?;
                let trhs: &mut [T] = tm(self.tmp2)?;
                let tdst: &mut [T] = tm(self.out)?;
                let clhs = conv_slice_dynamic(lhs, tlhs)?;
                let crhs = conv_slice_dynamic(rhs, trhs)?;
                let dst = bound_output_length(tdst, crhs)?;
                check_equal_lengths(clhs, crhs)?;
                check_ok_length(clhs)?;
                check_ok_length(crhs)?;
                check_ok_length(dst)?;
                let res = match op {
                    Add => <AddOp<T, T>>::apply_slice_slice(clhs, crhs, dst),
                    Sub => <SubOp<T, T>>::apply_slice_slice(clhs, crhs, dst),
                    Mul => <MulOp<T, T>>::apply_slice_slice(clhs, crhs, dst),
                    Div => <DivOp<T, T>>::apply_slice_slice(clhs, crhs, dst),
                    Rem => <RemOp<T, T>>::apply_slice_slice(clhs, crhs, dst),
                    Min => <MinOp<T, T>>::apply_slice_slice(clhs, crhs, dst),
                    Max => <MaxOp<T, T>>::apply_slice_slice(clhs, crhs, dst),
                    Pow => <PowOp<T, T>>::apply_slice_slice(clhs, crhs, dst),
                    BitAnd => <BitAndOp<T, T>>::apply_slice_slice(clhs, crhs, dst),
                    BitOr => <BitOrOp<T, T>>::apply_slice_slice(clhs, crhs, dst),
                    BitXor => <BitXorOp<T, T>>::apply_slice_slice(clhs, crhs, dst),
                };
                match res {
                    Ok(slice) => Ok(OperandSlice(slice.into())),
                    Err(_) => Err(EvalError::UnsupportedOp)
                }
            }
            (OperandSlice(lhs), OperandConst(rhs)) => {
                let tlhs: &mut [T] = tm(self.tmp1)?;
                let tdst: &mut [T] = tm(self.out)?;
                let clhs = conv_slice_dynamic(lhs, tlhs)?;
                let crhs = conv_const_dynamic(rhs)?;
                let dst = bound_output_length(tdst, clhs)?;
                check_ok_length(clhs)?;
                check_ok_length(dst)?;
                let res = match op {
                    Add => <AddOp<T, T>>::apply_slice_const(clhs, crhs, dst),
                    Sub => <SubOp<T, T>>::apply_slice_const(clhs, crhs, dst),
                    Mul => <MulOp<T, T>>::apply_slice_const(clhs, crhs, dst),
                    Div => <DivOp<T, T>>::apply_slice_const(clhs, crhs, dst),
                    Rem => <RemOp<T, T>>::apply_slice_const(clhs, crhs, dst),
                    Min => <MinOp<T, T>>::apply_slice_const(clhs, crhs, dst),
                    Max => <MaxOp<T, T>>::apply_slice_const(clhs, crhs, dst),
                    Pow => <PowOp<T, T>>::apply_slice_const(clhs, crhs, dst),
                    BitAnd => <BitAndOp<T, T>>::apply_slice_const(clhs, crhs, dst),
                    BitOr => <BitOrOp<T, T>>::apply_slice_const(clhs, crhs, dst),
                    BitXor => <BitXorOp<T, T>>::apply_slice_const(clhs, crhs, dst),
                };
                match res {
                    Ok(slice) => Ok(OperandSlice(slice.into())),
                    Err(_) => Err(EvalError::UnsupportedOp)
                }
            }
            (OperandConst(lhs), OperandSlice(rhs)) => {
                let trhs: &mut [T] = tm(self.tmp2)?;
                let tdst: &mut [T] = tm(self.out)?;
                let clhs = conv_const_dynamic(lhs)?;
                let crhs = conv_slice_dynamic(rhs, trhs)?;
                let dst = bound_output_length(tdst, crhs)?;
                check_ok_length(crhs)?;
                check_ok_length(dst)?;
                let res = match op {
                    Add => <AddOp<T, T>>::apply_const_slice(clhs, crhs, dst),
                    Sub => <SubOp<T, T>>::apply_const_slice(clhs, crhs, dst),
                    Mul => <MulOp<T, T>>::apply_const_slice(clhs, crhs, dst),
                    Div => <DivOp<T, T>>::apply_const_slice(clhs, crhs, dst),
                    Rem => <RemOp<T, T>>::apply_const_slice(clhs, crhs, dst),
                    Min => <MinOp<T, T>>::apply_const_slice(clhs, crhs, dst),
                    Max => <MaxOp<T, T>>::apply_const_slice(clhs, crhs, dst),
                    Pow => <PowOp<T, T>>::apply_const_slice(clhs, crhs, dst),
                    BitAnd => <BitAndOp<T, T>>::apply_const_slice(clhs, crhs, dst),
                    BitOr => <BitOrOp<T, T>>::apply_const_slice(clhs, crhs, dst),
                    BitXor => <BitXorOp<T, T>>::apply_const_slice(clhs, crhs, dst),
                };
                match res {
                    Ok(slice) => Ok(OperandSlice(slice.into())),
                    Err(_) => Err(EvalError::UnsupportedOp)
                }
            }
            (OperandConst(lhs), OperandConst(rhs)) => {
                let clhs = conv_const_dynamic(lhs)?;
                let crhs = conv_const_dynamic(rhs)?;
                let res = match op {
                    Add => <AddOp<T, T>>::apply_const_const(clhs, crhs),
                    Sub => <SubOp<T, T>>::apply_const_const(clhs, crhs),
                    Mul => <MulOp<T, T>>::apply_const_const(clhs, crhs),
                    Div => <DivOp<T, T>>::apply_const_const(clhs, crhs),
                    Rem => <RemOp<T, T>>::apply_const_const(clhs, crhs),
                    Min => <MinOp<T, T>>::apply_const_const(clhs, crhs),
                    Max => <MaxOp<T, T>>::apply_const_const(clhs, crhs),
                    Pow => <PowOp<T, T>>::apply_const_const(clhs, crhs),
                    BitAnd => <BitAndOp<T, T>>::apply_const_const(clhs, crhs),
                    BitOr => <BitOrOp<T, T>>::apply_const_const(clhs, crhs),
                    BitXor => <BitXorOp<T, T>>::apply_const_const(clhs, crhs),
                };
                match res {
                    Ok(c) => Ok(OperandConst(c.into())),
                    Err(_) => Err(EvalError::UnsupportedOp)
                }
            }
        }
    }

    #[inline(never)]
    fn bool_binop_static<'slice, T>(self, op: BoolBinOpCode,
                                    lhs: &Operand<'slice>,
                                    rhs: &Operand<'slice>)
                                    -> Result<Operand<'eval>, EvalError>
    where
        'slice: 'eval,
        T: 'eval,
        T: ScalarT,
        Slice<'slice>: From<&'eval [T]>,
        Const: From<T>,

        LtOp<T, bool>: BinOp<T, bool>,
        LeOp<T, bool>: BinOp<T, bool>,
        EqOp<T, bool>: BinOp<T, bool>,
        NeOp<T, bool>: BinOp<T, bool>,
        GeOp<T, bool>: BinOp<T, bool>,
        GtOp<T, bool>: BinOp<T, bool>,

        ConvOp<bool, T>: UnOp<bool, T>,
        ConvOp<u8, T>: UnOp<u8, T>,
        ConvOp<u16, T>: UnOp<u16, T>,
        ConvOp<u32, T>: UnOp<u32, T>,
        ConvOp<u64, T>: UnOp<u64, T>,
        ConvOp<u128, T>: UnOp<u128, T>,
        ConvOp<i8, T>: UnOp<i8, T>,
        ConvOp<i16, T>: UnOp<i16, T>,
        ConvOp<i32, T>: UnOp<i32, T>,
        ConvOp<i64, T>: UnOp<i64, T>,
        ConvOp<i128, T>: UnOp<i128, T>,
        ConvOp<f32, T>: UnOp<f32, T>,
        ConvOp<f64, T>: UnOp<f64, T>,
    {
        use Operand::*;
        use BoolBinOpCode::*;
        use transmute_buf_mut as tm;
        match (lhs, rhs) {
            (OperandSlice(lhs), OperandSlice(rhs)) => {
                let tlhs: &mut [T] = tm(self.tmp1)?;
                let trhs: &mut [T] = tm(self.tmp2)?;
                let tdst: &mut [bool] = tm(self.out)?;
                let clhs = conv_slice_dynamic(lhs, tlhs)?;
                let crhs = conv_slice_dynamic(rhs, trhs)?;
                let dst = bound_output_length(tdst, crhs)?;
                check_ok_length(clhs)?;
                check_ok_length(crhs)?;
                check_ok_length(dst)?;
                let res = match op {
                    Lt => <LtOp<T, bool>>::apply_slice_slice(clhs, crhs, dst),
                    Le => <LeOp<T, bool>>::apply_slice_slice(clhs, crhs, dst),
                    Eq => <EqOp<T, bool>>::apply_slice_slice(clhs, crhs, dst),
                    Ne => <NeOp<T, bool>>::apply_slice_slice(clhs, crhs, dst),
                    Ge => <GeOp<T, bool>>::apply_slice_slice(clhs, crhs, dst),
                    Gt => <GtOp<T, bool>>::apply_slice_slice(clhs, crhs, dst),
                };
                match res {
                    Ok(slice) => Ok(OperandSlice(slice.into())),
                    Err(_) => Err(EvalError::UnsupportedOp)
                }
            }
            (OperandSlice(lhs), OperandConst(rhs)) => {
                let tlhs: &mut [T] = tm(self.tmp1)?;
                let tdst: &mut [bool] = tm(self.out)?;
                let clhs = conv_slice_dynamic(lhs, tlhs)?;
                let crhs = conv_const_dynamic(rhs)?;
                let dst = bound_output_length(tdst, clhs)?;
                check_ok_length(clhs)?;
                check_ok_length(dst)?;
                let res = match op {
                    Lt => <LtOp<T, bool>>::apply_slice_const(clhs, crhs, dst),
                    Le => <LeOp<T, bool>>::apply_slice_const(clhs, crhs, dst),
                    Eq => <EqOp<T, bool>>::apply_slice_const(clhs, crhs, dst),
                    Ne => <NeOp<T, bool>>::apply_slice_const(clhs, crhs, dst),
                    Ge => <GeOp<T, bool>>::apply_slice_const(clhs, crhs, dst),
                    Gt => <GtOp<T, bool>>::apply_slice_const(clhs, crhs, dst),
                };
                match res {
                    Ok(slice) => Ok(OperandSlice(slice.into())),
                    Err(_) => Err(EvalError::UnsupportedOp)
                }
            }
            (OperandConst(lhs), OperandSlice(rhs)) => {
                let trhs: &mut [T] = tm(self.tmp2)?;
                let tdst: &mut [bool] = tm(self.out)?;
                let clhs = conv_const_dynamic(lhs)?;
                let crhs = conv_slice_dynamic(rhs, trhs)?;
                let dst = bound_output_length(tdst, crhs)?;
                check_ok_length(crhs)?;
                check_ok_length(dst)?;
                let res = match op {
                    Lt => <LtOp<T, bool>>::apply_const_slice(clhs, crhs, dst),
                    Le => <LeOp<T, bool>>::apply_const_slice(clhs, crhs, dst),
                    Eq => <EqOp<T, bool>>::apply_const_slice(clhs, crhs, dst),
                    Ne => <NeOp<T, bool>>::apply_const_slice(clhs, crhs, dst),
                    Ge => <GeOp<T, bool>>::apply_const_slice(clhs, crhs, dst),
                    Gt => <GtOp<T, bool>>::apply_const_slice(clhs, crhs, dst),
                };
                match res {
                    Ok(slice) => Ok(OperandSlice(slice.into())),
                    Err(_) => Err(EvalError::UnsupportedOp)
                }
            }
            (OperandConst(lhs), OperandConst(rhs)) => {
                let clhs = conv_const_dynamic(lhs)?;
                let crhs = conv_const_dynamic(rhs)?;
                let res = match op {
                    Lt => <LtOp<T, bool>>::apply_const_const(clhs, crhs),
                    Le => <LeOp<T, bool>>::apply_const_const(clhs, crhs),
                    Eq => <EqOp<T, bool>>::apply_const_const(clhs, crhs),
                    Ne => <NeOp<T, bool>>::apply_const_const(clhs, crhs),
                    Ge => <GeOp<T, bool>>::apply_const_const(clhs, crhs),
                    Gt => <GtOp<T, bool>>::apply_const_const(clhs, crhs),
                };
                match res {
                    Ok(c) => Ok(OperandConst(c.into())),
                    Err(_) => Err(EvalError::UnsupportedOp)
                }
            }
        }
    }

    #[inline(never)]
    fn val_unop_static<'slice, T>(self, op: ValUnOpCode,
                                  operand: &Operand<'slice>)
                                  -> Result<Operand<'eval>, EvalError>
    where
        'slice: 'eval,
        T: 'eval,
        T: ScalarT,
        Slice<'slice>: From<&'eval [T]>,
        Const: From<T>,

        NegOp<T, T>: UnOp<T, T>,
        AbsOp<T, T>: UnOp<T, T>,
        NotOp<T, T>: UnOp<T, T>,
        LnOp<T, T>: UnOp<T, T>,
        ExpOp<T, T>: UnOp<T, T>,
        SqrtOp<T, T>: UnOp<T, T>,
        SinOp<T, T>: UnOp<T, T>,
        CosOp<T, T>: UnOp<T, T>,

        ConvOp<bool, T>: UnOp<bool, T>,
        ConvOp<u8, T>: UnOp<u8, T>,
        ConvOp<u16, T>: UnOp<u16, T>,
        ConvOp<u32, T>: UnOp<u32, T>,
        ConvOp<u64, T>: UnOp<u64, T>,
        ConvOp<u128, T>: UnOp<u128, T>,
        ConvOp<i8, T>: UnOp<i8, T>,
        ConvOp<i16, T>: UnOp<i16, T>,
        ConvOp<i32, T>: UnOp<i32, T>,
        ConvOp<i64, T>: UnOp<i64, T>,
        ConvOp<i128, T>: UnOp<i128, T>,
        ConvOp<f32, T>: UnOp<f32, T>,
        ConvOp<f64, T>: UnOp<f64, T>,
    {
        use Operand::*;
        use ValUnOpCode::*;
        use transmute_buf_mut as tm;
        match operand {
            OperandSlice(s) => {
                let ts: &mut [T] = tm(self.tmp1)?;
                let tdst: &mut [T] = tm(self.out)?;
                let cs = conv_slice_dynamic(s, ts)?;
                let dst = bound_output_length(tdst, cs)?;
                check_ok_length(cs)?;
                check_ok_length(dst)?;
                let res = match op {
                    Neg => <NegOp<T, T>>::apply_slice(cs, dst),
                    BitNot => <NotOp<T, T>>::apply_slice(cs, dst),
                    Abs => <AbsOp<T, T>>::apply_slice(cs, dst),
                    Ln => <LnOp<T, T>>::apply_slice(cs, dst),
                    Exp => <ExpOp<T, T>>::apply_slice(cs, dst),
                    Sqrt => <SqrtOp<T, T>>::apply_slice(cs, dst),
                    Sin => <SinOp<T, T>>::apply_slice(cs, dst),
                    Cos => <CosOp<T, T>>::apply_slice(cs, dst),
                };
                match res {
                    Ok(slice) => Ok(OperandSlice(slice.into())),
                    Err(_) => Err(EvalError::UnsupportedOp)
                }
            }
            OperandConst(c) => {
                let cc = conv_const_dynamic(c)?;
                let res = match op {
                    Neg => <NegOp<T, T>>::apply_const(cc),
                    BitNot => <NotOp<T, T>>::apply_const(cc),
                    Abs => <AbsOp<T, T>>::apply_const(cc),
                    Ln => <LnOp<T, T>>::apply_const(cc),
                    Exp => <ExpOp<T, T>>::apply_const(cc),
                    Sqrt => <SqrtOp<T, T>>::apply_const(cc),
                    Sin => <SinOp<T, T>>::apply_const(cc),
                    Cos => <CosOp<T, T>>::apply_const(cc),
                };
                match res {
                    Ok(c) => Ok(OperandConst(c.into())),
                    Err(_) => Err(EvalError::UnsupportedOp)
                }
            }
        }
    }

    #[inline(never)]
    fn bool_unop_static<'slice, T>(self, op: BoolUnOpCode,
                                   operand: &Operand<'slice>)
                                    -> Result<Operand<'eval>, EvalError>
    where
        'slice: 'eval,
        T: 'eval,
        T: ScalarT,
        Slice<'slice>: From<&'eval [T]>,
        Const: From<T>,

        IsNaNOp<T, bool>: UnOp<T, bool>,
        IsInfOp<T, bool>: UnOp<T, bool>,
        IsFinOp<T, bool>: UnOp<T, bool>,

        ConvOp<bool, T>: UnOp<bool, T>,
        ConvOp<u8, T>: UnOp<u8, T>,
        ConvOp<u16, T>: UnOp<u16, T>,
        ConvOp<u32, T>: UnOp<u32, T>,
        ConvOp<u64, T>: UnOp<u64, T>,
        ConvOp<u128, T>: UnOp<u128, T>,
        ConvOp<i8, T>: UnOp<i8, T>,
        ConvOp<i16, T>: UnOp<i16, T>,
        ConvOp<i32, T>: UnOp<i32, T>,
        ConvOp<i64, T>: UnOp<i64, T>,
        ConvOp<i128, T>: UnOp<i128, T>,
        ConvOp<f32, T>: UnOp<f32, T>,
        ConvOp<f64, T>: UnOp<f64, T>,
    {
        use Operand::*;
        use BoolUnOpCode::*;
        use transmute_buf_mut as tm;
        match operand {
            OperandSlice(s) => {
                let ts: &mut [T] = tm(self.tmp1)?;
                let tdst: &mut [bool] = tm(self.out)?;
                let cs = conv_slice_dynamic(s, ts)?;
                let dst = bound_output_length(tdst, cs)?;
                check_ok_length(cs)?;
                check_ok_length(dst)?;
                let res = match op {
                    IsNaN => <IsNaNOp<T, bool>>::apply_slice(cs, dst),
                    IsInf => <IsInfOp<T, bool>>::apply_slice(cs, dst),
                    IsFin => <IsFinOp<T, bool>>::apply_slice(cs, dst),
                };
                match res {
                    Ok(slice) => Ok(OperandSlice(slice.into())),
                    Err(_) => Err(EvalError::UnsupportedOp)
                }
            }
            OperandConst(c) => {
                let cc = conv_const_dynamic(c)?;
                let res = match op {
                    IsNaN => <IsNaNOp<T, bool>>::apply_const(cc),
                    IsInf => <IsInfOp<T, bool>>::apply_const(cc),
                    IsFin => <IsFinOp<T, bool>>::apply_const(cc),
                };
                match res {
                    Ok(c) => Ok(OperandConst(c.into())),
                    Err(_) => Err(EvalError::UnsupportedOp)
                }
            }
        }
    }
}

fn conv_const_dynamic<DstT:ScalarT>(c: &Const) -> Result<DstT, EvalError>
where
    ConvOp<bool, DstT>: UnOp<bool, DstT>,
    ConvOp<u8, DstT>: UnOp<u8, DstT>,
    ConvOp<u16, DstT>: UnOp<u16, DstT>,
    ConvOp<u32, DstT>: UnOp<u32, DstT>,
    ConvOp<u64, DstT>: UnOp<u64, DstT>,
    ConvOp<u128, DstT>: UnOp<u128, DstT>,
    ConvOp<i8, DstT>: UnOp<i8, DstT>,
    ConvOp<i16, DstT>: UnOp<i16, DstT>,
    ConvOp<i32, DstT>: UnOp<i32, DstT>,
    ConvOp<i64, DstT>: UnOp<i64, DstT>,
    ConvOp<i128, DstT>: UnOp<i128, DstT>,
    ConvOp<f32, DstT>: UnOp<f32, DstT>,
    ConvOp<f64, DstT>: UnOp<f64, DstT>,
{
    use Const::*;
    let res = match *c {
        ConstBool(b) => <ConvOp<_, _>>::apply_const(b),
        ConstU8(u) => <ConvOp<_, _>>::apply_const(u),
        ConstU16(u) => <ConvOp<_, _>>::apply_const(u),
        ConstU32(u) => <ConvOp<_, _>>::apply_const(u),
        ConstU64(u) => <ConvOp<_, _>>::apply_const(u),
        ConstU128(u) => <ConvOp<_, _>>::apply_const(u),
        ConstI8(i) => <ConvOp<_, _>>::apply_const(i),
        ConstI16(i) => <ConvOp<_, _>>::apply_const(i),
        ConstI32(i) => <ConvOp<_, _>>::apply_const(i),
        ConstI64(i) => <ConvOp<_, _>>::apply_const(i),
        ConstI128(i) => <ConvOp<_, _>>::apply_const(i),
        ConstF32(v) => <ConvOp<_, _>>::apply_const(v),
        ConstF64(v) => <ConvOp<_, _>>::apply_const(v),
    };
    res.map_err(|_| EvalError::UnsupportedOp)
}

fn conv_slice_dynamic<'src, 'dst, DstT>(s: &Slice<'src>,
                                        tmp: &'dst mut [DstT])
                                        -> Result<&'dst [DstT], EvalError>
where
    'src: 'dst,
    DstT: 'dst,
    DstT: ScalarT,
    ConvOp<bool, DstT>: UnOp<bool, DstT>,
    ConvOp<u8, DstT>: UnOp<u8, DstT>,
    ConvOp<u16, DstT>: UnOp<u16, DstT>,
    ConvOp<u32, DstT>: UnOp<u32, DstT>,
    ConvOp<u64, DstT>: UnOp<u64, DstT>,
    ConvOp<u128, DstT>: UnOp<u128, DstT>,
    ConvOp<i8, DstT>: UnOp<i8, DstT>,
    ConvOp<i16, DstT>: UnOp<i16, DstT>,
    ConvOp<i32, DstT>: UnOp<i32, DstT>,
    ConvOp<i64, DstT>: UnOp<i64, DstT>,
    ConvOp<i128, DstT>: UnOp<i128, DstT>,
    ConvOp<f32, DstT>: UnOp<f32, DstT>,
    ConvOp<f64, DstT>: UnOp<f64, DstT>,
{
    use Slice::*;
    use bound_output_length as bl;
    let res = match *s {
        SliceBool(b) => <ConvOp<_, _>>::apply_slice(b, bl::<DstT, _>(tmp, b)?),
        SliceU8(u) => <ConvOp<_, _>>::apply_slice(u, bl::<DstT, _>(tmp, u)?),
        SliceU16(u) => <ConvOp<_, _>>::apply_slice(u, bl::<DstT, _>(tmp, u)?),
        SliceU32(u) => <ConvOp<_, _>>::apply_slice(u, bl::<DstT, _>(tmp, u)?),
        SliceU64(u) => <ConvOp<_, _>>::apply_slice(u, bl::<DstT, _>(tmp, u)?),
        SliceU128(u) => <ConvOp<_, _>>::apply_slice(u, bl::<DstT, _>(tmp, u)?),
        SliceI8(i) => <ConvOp<_, _>>::apply_slice(i, bl::<DstT, _>(tmp, i)?),
        SliceI16(i) => <ConvOp<_, _>>::apply_slice(i, bl::<DstT, _>(tmp, i)?),
        SliceI32(i) => <ConvOp<_, _>>::apply_slice(i, bl::<DstT, _>(tmp, i)?),
        SliceI64(i) => <ConvOp<_, _>>::apply_slice(i, bl::<DstT, _>(tmp, i)?),
        SliceI128(i) => <ConvOp<_, _>>::apply_slice(i, bl::<DstT, _>(tmp, i)?),
        SliceF32(v) => <ConvOp<_, _>>::apply_slice(v, bl::<DstT, _>(tmp, v)?),
        SliceF64(v) => <ConvOp<_, _>>::apply_slice(v, bl::<DstT, _>(tmp, v)?),
    };
    res.map_err(|_| EvalError::UnsupportedOp)
}
