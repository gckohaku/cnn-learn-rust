use std::marker::PhantomData;

use crate::nn_modules::{NNModule, NNModuleBuilder, NNModuleType, relu::ReLU};

pub struct ReLUBuilder<T> {
    _is_grad: bool,
    _phantom: PhantomData<T>,
}

impl<T> NNModuleBuilder<T> for ReLUBuilder<T>
where
    T: Clone,
    Box<dyn NNModule<T>>: Clone,
{
    type BuiltObject = NNModuleType<T>;

    fn new() -> Self {
        let is_grad = false;

        Self {
            _is_grad: is_grad,
            _phantom: PhantomData,
        }
    }

    fn build(self) -> NNModuleType<T> {
        NNModuleType::ReLU(ReLU::<T> {
            is_grad: self._is_grad,
            output_value: None,
            grad: None,
        })
    }
}

impl<T> ReLUBuilder<T> {
    pub fn is_grad(mut self, value: bool) -> Self {
        self._is_grad = value;
        self
    }
}
