use crate::nn_modules::{NNModuleBuilder, relu::ReLU};

pub struct ReLUBuilder {
	_inplace: bool,
}

impl NNModuleBuilder for ReLUBuilder {
	type BuiltObject = ReLU;

	fn new() -> Self {
		let inplace = false;

		Self {
			_inplace: inplace
		}
	}

	fn build(self) -> Self::BuiltObject {
		ReLU {
			inplace: self._inplace,
			output_value: None,
		}
	}
}

impl ReLUBuilder {
	pub fn inplace(mut self, value: bool) -> Self {
		self._inplace = value;
		self
	}
}