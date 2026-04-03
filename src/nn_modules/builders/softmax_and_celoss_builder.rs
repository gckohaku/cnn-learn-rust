use std::marker::PhantomData;

use crate::nn_modules::{
    NNModuleBuilder, NNNecessaryTraits, SoftmaxAndCELoss,
    builders::{CrossEntropyLossBuilder, SoftmaxBuilder},
};

pub struct SoftmaxAndCELossBuilder<T> {
    _is_test: bool,
    _phantom: PhantomData<T>,
}

impl<T> NNModuleBuilder<T> for SoftmaxAndCELossBuilder<T>
where
    T: NNNecessaryTraits,
{
    type BuiltObject = SoftmaxAndCELoss<T>;

    fn new() -> Self {
        let is_grad = false;

        Self {
            _is_test: is_grad,
            _phantom: PhantomData,
        }
    }

    fn build(self) -> Self::BuiltObject {
        let softmax = SoftmaxBuilder::new().build();
        let cross_entropy_loss = CrossEntropyLossBuilder::new().build();
        let test_result = (0usize, 0usize);

        SoftmaxAndCELoss::<T> {
            softmax,
            cross_entropy_loss,
            is_test: self._is_test,
            test_result,
            grad: None,
        }
    }
}

impl<T> SoftmaxAndCELossBuilder<T> {
    pub fn is_test(mut self, value: bool) -> Self {
        self._is_test = value;
        self
    }
}
