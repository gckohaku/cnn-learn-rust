pub trait HasZero {
	type Output;
	fn get_zero() -> Self::Output;
}

impl HasZero for f64 {
	type Output = f64;
	fn get_zero() -> f64 {
		0.0
	}
}

impl HasZero for f32 {
	type Output = f32;
	fn get_zero() -> f32 {
		0.0
	}
}