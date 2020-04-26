// Copyright 2019-2020 Graydon Hoare <graydon@pobox.com>
// Licensed under the MIT and Apache-2.0 licenses.

#[cfg(test)]
mod test_helpers {
    use crate::*;

    // We set this to a multiple of 16x CHUNKBYTES so that the worst case
    // (16-byte input type -> 1-byte output type) still passes an output vector
    // that's a multiple of CHUNKBYTES.
    const NBYTES: usize = CHUNKBYTES * 16 * 5;

    pub struct TestCtx {
        pub tmp1: Vec<u8>,
        pub tmp2: Vec<u8>,
        pub out: Vec<u8>
    }

    impl TestCtx {
        pub fn new() -> TestCtx {
            TestCtx {
                tmp1: vec![0; NBYTES],
                tmp2: vec![0; NBYTES],
                out: vec![0; NBYTES]
            }
        }
        pub fn len<T>(&self) -> usize {
            NBYTES / std::mem::size_of::<T>()
        }
        pub fn get_eval_ctx<'a>(&'a mut self) -> EvalCtx<'a> {
            EvalCtx {
                tmp1: &mut self.tmp1[..],
                tmp2: &mut self.tmp2[..],
                out:  &mut self.out[..]
            }
        }
    }
}


#[cfg(test)]
mod test_cmpops {
    use super::super::*;
    use super::test_helpers::*;
    use crate::ops::*;
    use crate::zeroone::*;
    use quickcheck::*;

    macro_rules! impl_test_bool_binop {
        ($T:ty, $($test:ident $opcode:ident $op:ident),*) => {
            $(
                #[test]
                fn $test() {
                    fn check_one(mut x: Vec<$T>, mut y: Vec<$T>) -> TestResult {
                        if x.len() == 0 {
                            x.push(<$T>::ONE)
                        }
                        if y.len() == 0 {
                            y.push(<$T>::ONE)
                        }
                        let mut tcx = TestCtx::new();
                        let a: Vec<$T> = x.iter().cloned().cycle().take(tcx.len::<$T>()).collect();
                        let b: Vec<$T> = y.iter().cloned().cycle().take(tcx.len::<$T>()).collect();
                        let exp: Vec<bool> = a.iter().zip(b.iter()).map(|(a, b)| a.$op(b)).collect();
                        let res = tcx.get_eval_ctx().bool_binop(BoolBinOpCode::$opcode,
                                                                &Operand::from(&a),
                                                                &Operand::from(&b));
                        match res {
                            Ok(r) => TestResult::from_bool(Operand::from(&exp) == r),
                            _ => TestResult::failed()
                        }
                    }
                    QuickCheck::new()
                        .tests(100)
                        .max_tests(100)
                        .quickcheck(check_one as fn(Vec<$T>,Vec<$T>) -> TestResult)
                }
            )*
        }
    }

    impl_test_bool_binop!(u8,
                          test_u8_lt Lt lt,
                          test_u8_le Le le,
                          test_u8_eq Eq eq,
                          test_u8_ne Ne ne,
                          test_u8_ge Ge ge,
                          test_u8_gt Gt gt);

    impl_test_bool_binop!(u16,
                          test_u16_lt Lt lt,
                          test_u16_le Le le,
                          test_u16_eq Eq eq,
                          test_u16_ne Ne ne,
                          test_u16_ge Ge ge,
                          test_u16_gt Gt gt);

    impl_test_bool_binop!(u32,
                          test_u32_lt Lt lt,
                          test_u32_le Le le,
                          test_u32_eq Eq eq,
                          test_u32_ne Ne ne,
                          test_u32_ge Ge ge,
                          test_u32_gt Gt gt);

    impl_test_bool_binop!(u64,
                          test_u64_lt Lt lt,
                          test_u64_le Le le,
                          test_u64_eq Eq eq,
                          test_u64_ne Ne ne,
                          test_u64_ge Ge ge,
                          test_u64_gt Gt gt);

    impl_test_bool_binop!(u128,
                          test_u128_lt Lt lt,
                          test_u128_le Le le,
                          test_u128_eq Eq eq,
                          test_u128_ne Ne ne,
                          test_u128_ge Ge ge,
                          test_u128_gt Gt gt);

    impl_test_bool_binop!(i8,
                          test_i8_lt Lt lt,
                          test_i8_le Le le,
                          test_i8_eq Eq eq,
                          test_i8_ne Ne ne,
                          test_i8_ge Ge ge,
                          test_i8_gt Gt gt);

    impl_test_bool_binop!(i16,
                          test_i16_lt Lt lt,
                          test_i16_le Le le,
                          test_i16_eq Eq eq,
                          test_i16_ne Ne ne,
                          test_i16_ge Ge ge,
                          test_i16_gt Gt gt);

    impl_test_bool_binop!(i32,
                          test_i32_lt Lt lt,
                          test_i32_le Le le,
                          test_i32_eq Eq eq,
                          test_i32_ne Ne ne,
                          test_i32_ge Ge ge,
                          test_i32_gt Gt gt);

    impl_test_bool_binop!(i64,
                          test_i64_lt Lt lt,
                          test_i64_le Le le,
                          test_i64_eq Eq eq,
                          test_i64_ne Ne ne,
                          test_i64_ge Ge ge,
                          test_i64_gt Gt gt);

    impl_test_bool_binop!(i128,
                          test_i128_lt Lt lt,
                          test_i128_le Le le,
                          test_i128_eq Eq eq,
                          test_i128_ne Ne ne,
                          test_i128_ge Ge ge,
                          test_i128_gt Gt gt);

    impl_test_bool_binop!(f32,
                          test_f32_lt Lt lt,
                          test_f32_le Le le,
                          test_f32_eq Eq eq,
                          test_f32_ne Ne ne,
                          test_f32_ge Ge ge,
                          test_f32_gt Gt gt);

    impl_test_bool_binop!(f64,
                          test_f64_lt Lt lt,
                          test_f64_le Le le,
                          test_f64_eq Eq eq,
                          test_f64_ne Ne ne,
                          test_f64_ge Ge ge,
                          test_f64_gt Gt gt);
}

#[cfg(test)]
mod test_valops {
    use super::super::*;
    use super::test_helpers::*;
    use crate::ops::*;
    use crate::zeroone::*;
    use std::ops::*;
    use quickcheck::*;

    macro_rules! impl_test_val_binop {
        ($T:ty, $($test:ident $opcode:ident $op:ident),*) => {
            $(
                #[test]
                fn $test() {
                    fn check_one(mut x: Vec<$T>, mut y: Vec<$T>) -> TestResult {
                        if x.len() == 0 {
                            x.push(<$T>::ONE)
                        }
                        if y.len() == 0 {
                            y.push(<$T>::ONE)
                        }
                        if ValBinOpCode::$opcode == ValBinOpCode::Div ||
                            ValBinOpCode::$opcode == ValBinOpCode::Rem {
                            for i in x.iter_mut() {
                                if *i == <$T>::ZERO {
                                    *i = <$T>::ONE
                                }
                            }
                            for i in y.iter_mut() {
                                if *i == <$T>::ZERO {
                                    *i = <$T>::ONE
                                }
                            }
                        }
                        let mut tcx = TestCtx::new();
                        let a: Vec<$T> = x.iter().cloned().cycle().take(tcx.len::<$T>()).collect();
                        let b: Vec<$T> = y.iter().cloned().cycle().take(tcx.len::<$T>()).collect();
                        let exp: Vec<$T> = a.iter().cloned().zip(b.iter().cloned()).map(|(a, b)| a.$op(b)).collect();
                        let res = tcx.get_eval_ctx().val_binop(ValBinOpCode::$opcode,
                                                               &Operand::from(&a),
                                                               &Operand::from(&b));
                        match res {
                            Ok(r) => TestResult::from_bool(Operand::from(&exp) == r),
                            _ => TestResult::failed()
                        }
                    }
                    QuickCheck::new()
                        .tests(100)
                        .max_tests(100)
                        .quickcheck(check_one as fn(Vec<$T>,Vec<$T>) -> TestResult)
                }
            )*
        }
    }

    impl_test_val_binop!(u8,
                         test_u8_add Add add,
                         test_u8_sub Sub sub,
                         test_u8_mul Mul mul,
                         test_u8_div Div div,
                         test_u8_rem Rem rem,
                         test_u8_min Min min,
                         test_u8_max Max max,
                         test_u8_bitand BitAnd bitand,
                         test_u8_bitor BitOr bitor,
                         test_u8_bitxor BitXor bitxor);

    impl_test_val_binop!(u16,
                         test_u16_add Add add,
                         test_u16_sub Sub sub,
                         test_u16_mul Mul mul,
                         test_u16_div Div div,
                         test_u16_rem Rem rem,
                         test_u16_min Min min,
                         test_u16_max Max max,
                         test_u16_bitand BitAnd bitand,
                         test_u16_bitor BitOr bitor,
                         test_u16_bitxor BitXor bitxor);

    impl_test_val_binop!(u32,
                         test_u32_add Add add,
                         test_u32_sub Sub sub,
                         test_u32_mul Mul mul,
                         test_u32_div Div div,
                         test_u32_rem Rem rem,
                         test_u32_min Min min,
                         test_u32_max Max max,
                         test_u32_bitand BitAnd bitand,
                         test_u32_bitor BitOr bitor,
                         test_u32_bitxor BitXor bitxor);

    impl_test_val_binop!(u64,
                         test_u64_add Add add,
                         test_u64_sub Sub sub,
                         test_u64_mul Mul mul,
                         test_u64_div Div div,
                         test_u64_rem Rem rem,
                         test_u64_min Min min,
                         test_u64_max Max max,
                         test_u64_bitand BitAnd bitand,
                         test_u64_bitor BitOr bitor,
                         test_u64_bitxor BitXor bitxor);

    impl_test_val_binop!(u128,
                         test_u128_add Add add,
                         test_u128_sub Sub sub,
                         test_u128_mul Mul mul,
                         test_u128_div Div div,
                         test_u128_rem Rem rem,
                         test_u128_min Min min,
                         test_u128_max Max max,
                         test_u128_bitand BitAnd bitand,
                         test_u128_bitor BitOr bitor,
                         test_u128_bitxor BitXor bitxor);

    impl_test_val_binop!(i8,
                         test_i8_add Add add,
                         test_i8_sub Sub sub,
                         test_i8_mul Mul mul,
                         test_i8_div Div div,
                         test_i8_rem Rem rem,
                         test_i8_min Min min,
                         test_i8_max Max max,
                         test_i8_bitand BitAnd bitand,
                         test_i8_bitor BitOr bitor,
                         test_i8_bitxor BitXor bitxor);

    impl_test_val_binop!(i16,
                         test_i16_add Add add,
                         test_i16_sub Sub sub,
                         test_i16_mul Mul mul,
                         test_i16_div Div div,
                         test_i16_rem Rem rem,
                         test_i16_min Min min,
                         test_i16_max Max max,
                         test_i16_bitand BitAnd bitand,
                         test_i16_bitor BitOr bitor,
                         test_i16_bitxor BitXor bitxor);

    impl_test_val_binop!(i32,
                         test_i32_add Add add,
                         test_i32_sub Sub sub,
                         test_i32_mul Mul mul,
                         test_i32_div Div div,
                         test_i32_rem Rem rem,
                         test_i32_min Min min,
                         test_i32_max Max max,
                         test_i32_bitand BitAnd bitand,
                         test_i32_bitor BitOr bitor,
                         test_i32_bitxor BitXor bitxor);

    impl_test_val_binop!(i64,
                         test_i64_add Add add,
                         test_i64_sub Sub sub,
                         test_i64_mul Mul mul,
                         test_i64_div Div div,
                         test_i64_rem Rem rem,
                         test_i64_min Min min,
                         test_i64_max Max max,
                         test_i64_bitand BitAnd bitand,
                         test_i64_bitor BitOr bitor,
                         test_i64_bitxor BitXor bitxor);

    impl_test_val_binop!(i128,
                         test_i128_add Add add,
                         test_i128_sub Sub sub,
                         test_i128_mul Mul mul,
                         test_i128_div Div div,
                         test_i128_rem Rem rem,
                         test_i128_min Min min,
                         test_i128_max Max max,
                         test_i128_bitand BitAnd bitand,
                         test_i128_bitor BitOr bitor,
                         test_i128_bitxor BitXor bitxor);
}
