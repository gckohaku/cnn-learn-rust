use std::marker::PhantomData;

use crate::nn_modules::{
    NNModuleBuilder, NNModuleType, NNNecessaryTraits, SoftmaxAndCELoss, builders::{CrossEntropyLossBuilder, SoftmaxBuilder}
};

pub struct SoftmaxAndCELossBuilder<T> {
    _phantom: PhantomData<T>,
}

impl<T> NNModuleBuilder<T> for SoftmaxAndCELossBuilder<T>
where T: NNNecessaryTraits
{
    type BuiltObject = NNModuleType<T>;

    fn new() -> Self {

        Self {
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
            grad: None,
        })
    }
}