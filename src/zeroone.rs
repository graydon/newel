// This code is copied from `const-identities` (MIT licensed);
// that crate depends on things I don't want to depend on,
// and doesn't support u128/i128.

pub trait ConstZero {
    const ZERO: Self;
}

pub trait ConstOne {
    const ONE: Self;
}

macro_rules! const_zero_one {
    (
        ($zero:expr, $one:expr) => [
            $( $T:ident ),* $(,)*
        ]
    ) => {
        $(
            impl ConstZero for $T {
                const ZERO: Self = $zero;
            }

            impl ConstOne for $T {
                const ONE: Self = $one;
            }
         )*
    }
}

const_zero_one! {
    (0, 1) => [
        i8, i16, i32, i64, i128, isize,
        u8, u16, u32, u64, u128, usize,
    ]
}

const_zero_one! {
    (0., 1.) => [f32, f64]
}
