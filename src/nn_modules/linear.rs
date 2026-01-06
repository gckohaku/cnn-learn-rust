use ndarray::{Array1, Array2};

use crate::nn_modules::NNModule;

/// アフィン変換を行うニューラルネットワークモジュール
pub struct Linear{
	pub weights: Array2<f64>,
	pub biases: Array1<f64>,
}

impl NNModule for Linear {
	type InputArray = Array2<f64>;
	type OutputArray = Array2<f64>;
	fn forward(&self, input: Array2<f64>) -> Array2<f64> {
		input.dot(&self.weights) + &self.biases
	}
}