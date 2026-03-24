use std::marker::PhantomData;

use crate::nn_modules::{NNModuleBuilder, NNModuleType, NNNecessaryTraits, relu::ReLU};

pub struct ReLUBuilder<T> {
    _phantom: PhantomData<T>,
}

impl<T> NNModuleBuilder<T> for ReLUBuilder<T>
where
    T: NNNecessaryTraits,
{
    type BuiltObject = NNModuleType<T>;

    fn new() -> Self {

        Self {
            _phantom: PhantomData,
        }
    }

    fn build(self) -> NNModuleType<T> {
        NNModuleType::ReLU(ReLU::<T> {
            output_value: None,
            grad: None,
        })
    }
}
