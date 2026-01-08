use crate::nn_modules::{CrossEntropyLoss, NNModuleBuilder};

pub struct CrossEntropyLossBuilder {}

impl NNModuleBuilder for CrossEntropyLossBuilder {
	type BuiltObject = CrossEntropyLoss;

	fn new() -> Self {
		CrossEntropyLossBuilder {  }
	}

	fn build(self) -> Self::BuiltObject {
		CrossEntropyLoss {
			expected_value: None,
		}
	}
}
