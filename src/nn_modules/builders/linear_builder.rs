use ndarray::{Array1, Array2};

use crate::nn_modules::linear::Linear;

struct LinearBuilder {
	input_node_value: usize,
	output_node_value: usize,
}

impl LinearBuilder {
	pub fn new() -> Self {
		let input_node_value = 0;
		let output_node_value = 0;

		LinearBuilder { input_node_value, output_node_value }
	}

	pub fn set_input_node_value(mut self, value: usize) -> Self {
		self.input_node_value = value;
		self
	}

	pub fn set_output_node_value(mut self, value: usize) -> Self {
		self.output_node_value = value;
		self
	}

	pub fn build(&self) -> Linear {
		let weights = Array2::<f64>::zeros((self.input_node_value, self.output_node_value));
		let biases = Array1::<f64>::zeros(self.input_node_value);

		Linear {
			weights,
			biases,
		}
	}
}