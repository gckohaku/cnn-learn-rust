use ndarray::{Array1, Array2};

use crate::nn_modules::{NNModuleBuilder, linear::Linear};

pub struct LinearBuilder {
	_input_node_value: usize,
	_output_node_value: usize,
}

impl NNModuleBuilder for LinearBuilder {
	type BuiltObject = Linear;

	fn new() -> Self {
		let input_node_value = 0;
		let output_node_value = 0;

		LinearBuilder { _input_node_value: input_node_value, _output_node_value: output_node_value }
	}

	fn build(self) -> Linear {
		let weights = Array2::<f64>::zeros((self._input_node_value, self._output_node_value));
		let biases = Array1::<f64>::zeros(self._input_node_value);

		Linear {
			weights,
			biases,
			input_value: None,
		}
	}
}

impl LinearBuilder {
	pub fn input_node_value(mut self, value: usize) -> Self {
		self._input_node_value = value;
		self
	}

	pub fn output_node_value(mut self, value: usize) -> Self {
		self._output_node_value = value;
		self
	}
}