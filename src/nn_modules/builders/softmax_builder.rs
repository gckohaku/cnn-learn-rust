use std::marker::PhantomData;

use crate::nn_modules::{NNModuleBuilder, NNNecessaryTraits, Softmax};

pub struct SoftmaxBuilder<T> {
    _is_grad: bool,
    _phantom: PhantomData<T>,
}

impl<T> NNModuleBuilder<T> for SoftmaxBuilder<T>
where
    T: NNNecessaryTraits,
{
    type BuiltObject = Softmax<T>;

    fn new() -> Self {
        let is_grad = false;

        SoftmaxBuilder::<T> {
            _is_grad: is_grad,
            _phantom: PhantomData,
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

impl<T> SoftmaxBuilder<T> {
    pub fn is_grad(mut self, value: bool) -> Self {
        self._is_grad = value;
        self
    }
}
