use std::marker::PhantomData;

use crate::nn_modules::{NNModuleBuilder, NNModuleType, NNNecessaryTraits, ReshapeTensor};

pub struct ReshapeTensorBuilder<T> {
    _shape: Vec<usize>,
    _phantom: PhantomData<T>,
}

impl<T> NNModuleBuilder<T> for ReshapeTensorBuilder<T>
where
    T: NNNecessaryTraits,
{
    type BuiltObject = NNModuleType<T>;

    fn new() -> Self {
        let _shape = Vec::<usize>::new();

        Self {
            _shape,
            _phantom: PhantomData,
        }
    }

    fn build(self) -> Self::BuiltObject {
        let shape = self._shape;

        NNModuleType::ReshapeTensor(ReshapeTensor::<T> {
            shape,
            before_shape: None,
            _phantom: PhantomData,
        })
    }
}

impl<T> ReshapeTensorBuilder<T> {
    pub fn shape(mut self, shape: Vec<usize>) -> Self {
        self._shape = shape;
        self
    }
}
