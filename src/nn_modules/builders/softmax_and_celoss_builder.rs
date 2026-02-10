use std::marker::PhantomData;

use crate::nn_modules::{
    NNModuleBuilder, SoftmaxAndCELoss,
    builders::{CrossEntropyLossBuilder, SoftmaxBuilder},
    cross_entropy_loss, softmax,
};


pub struct SoftmaxAndCELossBuilder<T> {
    _is_grad: bool,
    _phantom: PhantomData<T>,
}

impl<T> NNModuleBuilder<T> for SoftmaxAndCELossBuilder<T> {
    type BuiltObject = SoftmaxAndCELoss<T>;

    fn new() -> Self {
        let is_grad = false;

        Self {
            _is_grad: is_grad,
            _phantom: PhantomData,
        }
    }

    fn build(self) -> Self::BuiltObject {
        let softmax = SoftmaxBuilder::new().build();
        let cross_entropy_loss = CrossEntropyLossBuilder::new().build();

        SoftmaxAndCELoss::<T> {
            softmax,
            cross_entropy_loss,
            is_grad: self._is_grad,
            grad: None,
        }
    }
}

impl<T> SoftmaxAndCELossBuilder<T> {
    fn is_grad(mut self, value: bool) -> Self {
        self._is_grad = value;
        self
    }
}
