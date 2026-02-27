use std::marker::PhantomData;

use crate::nn_modules::{CrossEntropyLoss, NNModule, NNModuleBuilder, NNModuleType};

pub struct CrossEntropyLossBuilder<T> {
    _is_grad: bool,
    _marker: PhantomData<T>,
}

impl<T> NNModuleBuilder<T> for CrossEntropyLossBuilder<T>
where
{
    type BuiltObject = NNModuleType<T>;

    fn new() -> Self {
        let is_grad = false;

        CrossEntropyLossBuilder::<T> {
            _is_grad: is_grad,
            _marker: PhantomData,
        }
    }

    fn build(self) -> NNModuleType<T> {
        NNModuleType::CrossEntropyLoss(CrossEntropyLoss {
            is_grad: self._is_grad,
            expected_value: None,
        })
    }
}

impl<T> CrossEntropyLossBuilder<T> {
    pub fn is_grad(mut self, value: bool) -> Self {
        self._is_grad = value;
        self
    }
}
