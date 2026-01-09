use crate::nn_modules::{NNModuleBuilder, relu::ReLU};

pub struct ReLUBuilder {
    _inplace: bool,
    _is_grad: bool,
}

impl NNModuleBuilder for ReLUBuilder {
    type BuiltObject = ReLU;

    fn new() -> Self {
        let inplace = false;
        let is_grad = false;

        Self {
            _inplace: inplace,
            _is_grad: is_grad,
        }
    }

    fn build(self) -> Self::BuiltObject {
        ReLU {
            inplace: self._inplace,
            is_grad: self._is_grad,
            output_value: None,
            grad: None,
        }
    }
}

impl ReLUBuilder {
    pub fn inplace(mut self, value: bool) -> Self {
        self._inplace = value;
        self
    }

    pub fn is_grad(mut self, value: bool) -> Self {
        self._is_grad = value;
        self
    }
}
