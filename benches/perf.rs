// Copyright 2019-2020 Graydon Hoare <graydon@pobox.com>
// Licensed under the MIT and Apache-2.0 licenses.

#[macro_use]
extern crate bencher;
extern crate newel;

use bencher::Bencher;
use newel::*;

pub struct BenchCtx {
    pub tmp1: Vec<u8>,
    pub tmp2: Vec<u8>,
    pub out: Vec<u8>
}

impl BenchCtx {
    pub fn new() -> BenchCtx {
        BenchCtx {
            tmp1: vec![0; 0x1000000],
            tmp2: vec![0; 0x1000000],
            out: vec![0; 0x1000000]
        }
    }
    pub fn get_eval_ctx<'a>(&'a mut self) -> EvalCtx<'a> {
        EvalCtx {
            tmp1: &mut self.tmp1[..],
            tmp2: &mut self.tmp2[..],
            out:  &mut self.out[..]
        }
    }
}

#[inline(never)]
fn add_u8(a: &u8, b: &u8, c: &mut u8) {
    *c = *a + *b;
}

#[inline(never)]
fn add_u32(a: &u32, b: &u32, c: &mut u32) {
    *c = *a + *b;
}

#[inline(never)]
fn add_u64(a: &u64, b: &u64, c: &mut u64) {
    *c = *a + *b;
}

#[inline(never)]
fn lt_u8(a: &u8, b: &u8, c: &mut bool) {
    *c = *a < *b;
}

#[inline(never)]
fn lt_u16(a: &u16, b: &u16, c: &mut bool) {
    *c = *a < *b;
}

#[inline(never)]
fn lt_u32(a: &u32, b: &u32, c: &mut bool) {
    *c = *a < *b;
}

#[inline(never)]
fn get_bool_testvec() -> Vec<bool> {
    (0..0x100000).map(|_| false).collect()
}

#[inline(never)]
fn get_u8_testvec(scale: u8) -> Vec<u8> {
    (0..0x100000u32).map(|x| (x as u8) * scale).collect()
}

#[inline(never)]
fn get_u16_testvec(scale: u16) -> Vec<u16> {
    (0..0x100000u32).map(|x| (x as u16) * scale).collect()
}

#[inline(never)]
fn get_u32_testvec(scale: u32) -> Vec<u32> {
    (0..0x100000).map(|x| x * scale).collect()
}

#[inline(never)]
fn get_u64_testvec(scale: u64) -> Vec<u64> {
    (0..0x100000).map(|x| x * scale).collect()
}

fn add_u8_scalar(bench: &mut Bencher) {
    let a: Vec<u8> = get_u8_testvec(1);
    let b: Vec<u8> = get_u8_testvec(2);
    let mut c: Vec<u8> = get_u8_testvec(0);
    bench.iter(|| {
        for ((av, bv), cv) in a.iter().zip(b.iter()).zip(c.iter_mut()) {
            add_u8(av, bv, cv)
        }
    });
    bench.bytes = (a.len() as u64) * (::std::mem::size_of::<u8>() as u64)
}

fn add_u8_newel(bench: &mut Bencher) {
    let a: Vec<u8> = get_u8_testvec(1);
    let b: Vec<u8> = get_u8_testvec(2);
    let mut bcx = BenchCtx::new();
    bench.iter(|| {
        let res = bcx.get_eval_ctx().val_binop(ValBinOpCode::Add,
                                               &Operand::from(&a),
                                               &Operand::from(&b));
        assert!(res.is_ok());
    });
    bench.bytes = (a.len() as u64) * (::std::mem::size_of::<u8>() as u64)
}

fn add_u32_scalar(bench: &mut Bencher) {
    let a: Vec<u32> = get_u32_testvec(1);
    let b: Vec<u32> = get_u32_testvec(2);
    let mut c: Vec<u32> = get_u32_testvec(0);
    bench.iter(|| {
        for ((av, bv), cv) in a.iter().zip(b.iter()).zip(c.iter_mut()) {
            add_u32(av, bv, cv)
        }
    });
    bench.bytes = (a.len() as u64) * (::std::mem::size_of::<u32>() as u64)
}

fn add_u32_newel(bench: &mut Bencher) {
    let a: Vec<u32> = get_u32_testvec(1);
    let b: Vec<u32> = get_u32_testvec(2);
    let mut bcx = BenchCtx::new();
    bench.iter(|| {
        let res = bcx.get_eval_ctx().val_binop(ValBinOpCode::Add,
                                               &Operand::from(&a),
                                               &Operand::from(&b));
        assert!(res.is_ok());
    });
    bench.bytes = (a.len() as u64) * (::std::mem::size_of::<u32>() as u64)
}

fn add_u64_scalar(bench: &mut Bencher) {
    let a: Vec<u64> = get_u64_testvec(1);
    let b: Vec<u64> = get_u64_testvec(2);
    let mut c: Vec<u64> = get_u64_testvec(0);
    bench.iter(|| {
        for ((av, bv), cv) in a.iter().zip(b.iter()).zip(c.iter_mut()) {
            add_u64(av, bv, cv)
        }
    });
    bench.bytes = (a.len() as u64) * (::std::mem::size_of::<u64>() as u64)
}

fn add_u64_newel(bench: &mut Bencher) {
    let a: Vec<u64> = get_u64_testvec(1);
    let b: Vec<u64> = get_u64_testvec(2);
    let mut bcx = BenchCtx::new();
    bench.iter(|| {
        let res = bcx.get_eval_ctx().val_binop(ValBinOpCode::Add,
                                               &Operand::from(&a),
                                               &Operand::from(&b));
        assert!(res.is_ok());
    });
    bench.bytes = (a.len() as u64) * (::std::mem::size_of::<u64>() as u64)
}

fn lt_u8_scalar(bench: &mut Bencher) {
    let a: Vec<u8> = get_u8_testvec(1);
    let b: Vec<u8> = get_u8_testvec(2);
    let mut c: Vec<bool> = get_bool_testvec();
    bench.iter(|| {
        for ((av, bv), cv) in a.iter().zip(b.iter()).zip(c.iter_mut()) {
            lt_u8(av, bv, cv)
        }
    });
    bench.bytes = (a.len() as u64) * (::std::mem::size_of::<u8>() as u64)
}

fn lt_u8_newel(bench: &mut Bencher) {
    let a: Vec<u8> = get_u8_testvec(1);
    let b: Vec<u8> = get_u8_testvec(2);
    let mut bcx = BenchCtx::new();
    bench.iter(|| {
        let res = bcx.get_eval_ctx().bool_binop(BoolBinOpCode::Lt,
                                                &Operand::from(&a),
                                                &Operand::from(&b));
        assert!(res.is_ok());
    });
    bench.bytes = (a.len() as u64) * (::std::mem::size_of::<u8>() as u64)
}

fn lt_u16_scalar(bench: &mut Bencher) {
    let a: Vec<u16> = get_u16_testvec(1);
    let b: Vec<u16> = get_u16_testvec(2);
    let mut c: Vec<bool> = get_bool_testvec();
    bench.iter(|| {
        for ((av, bv), cv) in a.iter().zip(b.iter()).zip(c.iter_mut()) {
            lt_u16(av, bv, cv)
        }
    });
    bench.bytes = (a.len() as u64) * (::std::mem::size_of::<u16>() as u64)
}

fn lt_u16_newel(bench: &mut Bencher) {
    let a: Vec<u16> = get_u16_testvec(1);
    let b: Vec<u16> = get_u16_testvec(2);
    let mut bcx = BenchCtx::new();
    bench.iter(|| {
        let res = bcx.get_eval_ctx().bool_binop(BoolBinOpCode::Lt,
                                                &Operand::from(&a),
                                                &Operand::from(&b));
        assert!(res.is_ok());
    });
    bench.bytes = (a.len() as u64) * (::std::mem::size_of::<u16>() as u64)
}

fn lt_u32_scalar(bench: &mut Bencher) {
    let a: Vec<u32> = get_u32_testvec(1);
    let b: Vec<u32> = get_u32_testvec(2);
    let mut c: Vec<bool> = get_bool_testvec();
    bench.iter(|| {
        for ((av, bv), cv) in a.iter().zip(b.iter()).zip(c.iter_mut()) {
            lt_u32(av, bv, cv)
        }
    });
    bench.bytes = (a.len() as u64) * (::std::mem::size_of::<u32>() as u64)
}

fn lt_u32_newel(bench: &mut Bencher) {
    let a: Vec<u32> = get_u32_testvec(1);
    let b: Vec<u32> = get_u32_testvec(2);
    let mut bcx = BenchCtx::new();
    bench.iter(|| {
        let res = bcx.get_eval_ctx().bool_binop(BoolBinOpCode::Lt,
                                                &Operand::from(&a),
                                                &Operand::from(&b));
        assert!(res.is_ok());
    });
    bench.bytes = (a.len() as u64) * (::std::mem::size_of::<u32>() as u64)
}

benchmark_group!(
    benches,
    add_u8_scalar,
    add_u8_newel,
    add_u32_scalar,
    add_u32_newel,
    add_u64_scalar,
    add_u64_newel,
    lt_u8_scalar,
    lt_u8_newel,
    lt_u16_scalar,
    lt_u16_newel,
    lt_u32_scalar,
    lt_u32_newel
);
benchmark_main!(benches);
