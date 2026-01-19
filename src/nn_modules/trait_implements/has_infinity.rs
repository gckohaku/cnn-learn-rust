use core::f64;

pub trait HasInfinity {
	type Output;
	fn get_infinity() -> Self::Output;
	fn get_neg_infinity() -> Self::Output;
}

impl HasInfinity for f64 {
	type Output = f64;
	#[inline]
	fn get_infinity() -> Self::Output {
		f64::INFINITY
	}

	#[inline]
	fn get_neg_infinity() -> Self::Output {
		f64::NEG_INFINITY
	}
}

impl HasInfinity for f32 {
	type Output = f32;
	#[inline]
	fn get_infinity() -> Self::Output {
		f32::INFINITY
	}

	#[inline]
	fn get_neg_infinity() -> Self::Output {
		f32::NEG_INFINITY
	}
}