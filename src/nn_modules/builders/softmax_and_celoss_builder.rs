use std::marker::PhantomData;

use crate::nn_modules::{
    NNModule, NNModuleBuilder, NNModuleType, Softmax, SoftmaxAndCELoss, builders::{CrossEntropyLossBuilder, SoftmaxBuilder}, cross_entropy_loss
};

pub struct SoftmaxAndCELossBuilder<T> {
    _is_grad: bool,
    _phantom: PhantomData<T>,
}

impl<T> NNModuleBuilder<T> for SoftmaxAndCELossBuilder<T>
where
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

    fn build(self) -> Self::BuiltObject {
        let softmax_enum_data = SoftmaxBuilder::new().build();
        let cross_entropy_loss_enum_data = CrossEntropyLossBuilder::new().build();

        let softmax = match softmax_enum_data {
            NNModuleType::Softmax(s) => s,
            _ => panic!("invalid type accept"),
        };

        let cross_entropy_loss = match cross_entropy_loss_enum_data {
            NNModuleType::CrossEntropyLoss(s) => s,
            _ => panic!("invalid type accept"),
        };

        NNModuleType::SoftmaxAndCELoss(SoftmaxAndCELoss::<T> {
            softmax,
            cross_entropy_loss,
            is_grad: self._is_grad,
            grad: None,
        })
    }
}

impl<T> SoftmaxAndCELossBuilder<T> {
    fn is_grad(mut self, value: bool) -> Self {
        self._is_grad = value;
        self
    }
}
