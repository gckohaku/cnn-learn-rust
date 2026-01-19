pub trait HasOne {
	type Output;
	fn get_one() -> Self::Output;
}

impl HasOne for f64 {
	type Output = f64;
	#[inline]
	fn get_one() -> f64 {
		1.0
	}
}

impl HasOne for f32 {
	type Output = f32;
	#[inline]
	fn get_one() -> f32 {
		1.0
	}
}