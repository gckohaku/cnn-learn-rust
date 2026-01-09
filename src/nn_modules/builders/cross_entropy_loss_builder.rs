use crate::nn_modules::{CrossEntropyLoss, NNModuleBuilder};

pub struct CrossEntropyLossBuilder {
    _is_grad: bool,
}

impl NNModuleBuilder for CrossEntropyLossBuilder {
    type BuiltObject = CrossEntropyLoss;

    fn new() -> Self {
        let is_grad = false;

        CrossEntropyLossBuilder {
            _is_grad: is_grad,
        }
    }

    fn build(self) -> Self::BuiltObject {
        CrossEntropyLoss {
			is_grad: self._is_grad,
            expected_value: None,
        }
    }
}

impl CrossEntropyLossBuilder {
	pub fn is_grad(mut self, value: bool) -> Self {
        self._is_grad = value;
        self
    }
}
