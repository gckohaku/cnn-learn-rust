use std::marker::PhantomData;

use crate::nn_modules::{NNModuleBuilder, relu::ReLU};

pub struct ReLUBuilder<T> {
    _is_grad: bool,
    _phantom: PhantomData<T>
}

impl<T> NNModuleBuilder<T> for ReLUBuilder<T> where T: Clone {
    type BuiltObject = ReLU<T>;

    fn new() -> Self {
        let is_grad = false;

        Self {
            _is_grad: is_grad,
            _phantom: PhantomData,
        }
    }

    fn build(self) -> Self::BuiltObject {
        ReLU::<T> {
            is_grad: self._is_grad,
            output_value: None,
            grad: None,
        }
    }
}

impl<T> ReLUBuilder<T> {
    pub fn is_grad(mut self, value: bool) -> Self {
        self._is_grad = value;
        self
    }
}
