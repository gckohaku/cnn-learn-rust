use std::marker::PhantomData;

use crate::{
    cnn_information::PoolingType,
    nn_modules::{NNModuleBuilder, NNNecessaryTraits, Pooling},
};

pub struct PoolingBuilder<T> {
    _pooling_type: PoolingType,
    _window_size: (usize, usize),
    _stride: usize,
    _phantom: PhantomData<T>,
}

impl<T> NNModuleBuilder<T> for PoolingBuilder<T>
where
    T: NNNecessaryTraits,
{
    type BuiltObject = Pooling<T>;

    fn new() -> Self {
        let _pooling_type = PoolingType::MaxPooling;
        let _window_size = (1usize, 1usize);
        let _stride = 1usize;

        PoolingBuilder::<T> {
            _pooling_type,
            _window_size,
            _stride,
            _phantom: PhantomData,
        }
    }

    fn build(self) -> Self::BuiltObject {
        let pooling_type = self._pooling_type;
        let window_size = self._window_size;
        let stride = self._stride;

        Pooling::<T> {
            pooling_type,
            window_size,
            stride,
            pooling_mask: None,
            _phantom: PhantomData,
        }
    }
}

impl<T> PoolingBuilder<T> {
    pub fn pooling_type(mut self, t: PoolingType) -> Self {
        self._pooling_type = t;
        self
    }

    pub fn window_size(mut self, size: (usize, usize)) -> Self {
        self._window_size = size;
        self
    }

    pub fn stride(mut self, value: usize) -> Self {
        self._stride = value;
        self
    }
}
