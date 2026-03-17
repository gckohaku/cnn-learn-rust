use std::{
    f64::consts::{self},
    time::{SystemTime, UNIX_EPOCH},
};

use num_traits::{ConstOne, ConstZero, Float, Num};

// PCD の XSL-RR での疑似乱数生成
#[derive(Clone, Debug)]
pub struct Rand {
    state: u128,
}

const MULTIPLIER: u128 = 0x7a5cd86ea68452e3f784a651a38c59b1;
const ADDER: u128 = 0x2a4e8f94a486e68423bc84d7856d1323;

fn rotl64(x: u64, k: u8) -> u64 {
    (&x << &k) | (&x >> (64 - &k))
}

fn rotr64(x: u64, k: u8) -> u64 {
    rotl64(x, 64 - k)
}

fn generate_state_u64(seed: u64) -> u128 {
    let top_quad_word = u128::from(seed ^ rotr64(seed, 5)) << 64;
    let bottom_quad_word = u128::from(seed);

    top_quad_word | bottom_quad_word
}

impl Rand {
    pub fn new() -> Self {
        let seed: u64 = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Rand::new_with_seed_u64(seed)
    }

    pub fn new_with_seed_u64(seed: u64) -> Self {
        let mut r = Self {
            state: generate_state_u64(seed),
        };
        r.next();
        r
    }

    pub fn set_seed_u64(&mut self, seed: u64) {
        let top_quad_word = u128::from(seed ^ rotr64(seed, 5)) << 64;
        let bottom_quad_word = u128::from(seed);

        self.state = generate_state_u64(seed);
    }

    pub fn set_seed_u128(&mut self, seed: u128) {
        self.state = seed;
    }

    pub fn next(&mut self) -> u64 {
        let before_state = self.state;
        let count: u8 = (self.state >> 122) as u8;
        let x_no_process = self.state;

        self.state = (x_no_process.wrapping_mul(MULTIPLIER)).wrapping_add(ADDER);

        let x: u64 = (x_no_process ^ (x_no_process >> 64)) as u64;

        rotr64(x, count)
    }

    /// min 以上 max 未満のランダムな u32 を返す
    pub fn rand_u32_range(&mut self, min: u32, max: u32) -> u32 {
        let range = max - min;

        let mut value = self.next() as u32;
        loop {
            if value / range >= 0xffffffff / range {
                value = self.next() as u32;
                continue;
            }

            break;
        }

        (value % range) + min
    }

    /// min 以上 max 未満のランダムな usize を返す
    pub fn rand_usize_range(&mut self, min: usize, max: usize) -> usize {
        let range = max - min;

        let mut value = self.next() as usize;
        // println!("{}", self.next());
        loop {
            if value / range >= usize::MAX / range {
                value = self.next() as usize;
                continue;
            }

            break;
        }

        (value % range) + min
    }

    // TODO: 最大値と最小値を指定した乱数、少数での乱数の関数を作成する

    /// 0 以上 1 未満の乱数を返す
    pub fn rand_f64(&mut self) -> f64 {
        let exponent_bias: i64 = 1023;
        let mut ret_u64 = self.next() & 0x1fffffffffffff;

        if ret_u64 == 0 {
            return 0.0;
        }

        let mut exponent: i64 = 0;

        loop {
            ret_u64 <<= 1;
            exponent -= 1;

            if ret_u64 & 0x10000000000000 == 0x10000000000000 {
                break;
            }
        }

        exponent += exponent_bias;

        let ret_f64: f64 = f64::from_bits(
            (((exponent as u64) << 52) & 0x7ff0000000000000) | (ret_u64 & 0xfffffffffffff),
        );

        ret_f64
    }

    pub fn rand<T>(&mut self) -> T
    where
        T: Num + Float + ConstZero,
    {
        let exponent_bias: i64 = 1023;
        let mut ret_u64 = self.next() & 0x1fffffffffffff;

        if ret_u64 == 0 {
            return T::ZERO;
        }

        let mut exponent: i64 = 0;

        loop {
            ret_u64 <<= 1;
            exponent -= 1;

            if ret_u64 & 0x10000000000000 == 0x10000000000000 {
                break;
            }
        }

        exponent += exponent_bias;

        let ret_f64: f64 = f64::from_bits(
            (((exponent as u64) << 52) & 0x7ff0000000000000) | (ret_u64 & 0xfffffffffffff),
        );

        T::from(ret_f64).expect("Cast from f64 is failed")
    }

    /// 平均 `mu` 、分散 `sigma` の正規分布に従う乱数を生成する
    ///
    /// 通常の乱数は一様分布に従うような形になっているが、一様分布では不都合がある時に使う
    pub fn normal<T>(&mut self, mu: T, sigma: T) -> T
    where
        T: Num + Float + ConstZero + ConstOne,
    {
        self.normal_use_cosine(mu, sigma)
    }

    pub fn normal_use_cosine<T>(&mut self, mu: T, sigma: T) -> T
    where
        T: Num + Float + ConstZero + ConstOne,
    {
        let x = self.rand::<T>();
        let y = self.rand::<T>();

        let two = T::ONE + T::ONE;

        let z = (-two * x.ln()).sqrt()
            * (two * T::from(consts::PI).expect("PI cannot to cast from f64 to T") * y).cos();

        mu + z * sigma
    }

    pub fn normal_use_sine(&mut self, mu: f64, sigma: f64) -> f64 {
        let x = self.rand_f64();
        let y = self.rand_f64();

        let z = (-2.0 * x.ln()).sqrt() * (2.0 * consts::PI * y).sin();

        mu + z * sigma
    }
}
