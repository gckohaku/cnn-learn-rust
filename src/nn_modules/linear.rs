use ndarray::{Array1, Array2};

use crate::nn_modules::NNModule;

pub struct Linear{
	pub weights: Array2<f64>,
	pub biases: Array1<f64>,
}

impl Linear for NNModule {
	fn forward<T, R>(tensor: T) -> R {

	}
}