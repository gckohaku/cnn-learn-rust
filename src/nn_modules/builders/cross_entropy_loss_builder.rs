use std::marker::PhantomData;

use crate::nn_modules::{CrossEntropyLoss, NNModuleBuilder, NNModuleType, NNNecessaryTraits};

pub struct CrossEntropyLossBuilder<T> {
    _marker: PhantomData<T>,
}

impl<T> NNModuleBuilder<T> for CrossEntropyLossBuilder<T>
where T: NNNecessaryTraits
{
    type BuiltObject = NNModuleType<T>;

    fn new() -> Self {
        CrossEntropyLossBuilder::<T> {
            _marker: PhantomData,
        }
    }

    fn build(self) -> NNModuleType<T> {
        NNModuleType::CrossEntropyLoss(CrossEntropyLoss {
            expected_value: None,
        })
    }
}
