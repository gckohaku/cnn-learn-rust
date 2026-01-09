use crate::nn_modules::{NNModuleBuilder, SoftmaxAndCELoss, builders::{CrossEntropyLossBuilder, SoftmaxBuilder}, cross_entropy_loss, softmax};

pub struct SoftmaxAndCELossBuilder {
	_is_grad: bool,
}

impl NNModuleBuilder for SoftmaxAndCELossBuilder {
	type BuiltObject = SoftmaxAndCELoss;

	fn new() -> Self {
		let is_grad = false;

		Self { _is_grad: is_grad }
	}

	fn build(self) -> Self::BuiltObject {
		let softmax = SoftmaxBuilder::new().build();
		let cross_entropy_loss = CrossEntropyLossBuilder::new().build();

		SoftmaxAndCELoss {
			softmax,
			cross_entropy_loss,
			is_grad: self._is_grad,
			grad: None
		}
	}
}

impl SoftmaxAndCELossBuilder {
	fn is_grad(mut self, value: bool) -> Self {
		self._is_grad = value;
		self
	}
}