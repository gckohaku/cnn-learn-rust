use crate::nn_modules::{NNModuleBuilder, Softmax};

pub struct SoftmaxBuilder {
    _is_grad: bool,
}

impl NNModuleBuilder for SoftmaxBuilder {
    type BuiltObject = Softmax;

    fn new() -> Self {
        let is_grad = false;

        SoftmaxBuilder {
            _is_grad: is_grad,
        }
    }

    fn build(self) -> Self::BuiltObject {
        Softmax {
            is_grad: self._is_grad,
            output_value: None,
            grad: None,
        }
    }
}

impl SoftmaxBuilder {
    pub fn is_grad(mut self, value: bool) -> Self {
        self._is_grad = value;
        self
    }
}