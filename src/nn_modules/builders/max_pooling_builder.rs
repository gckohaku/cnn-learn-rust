use std::marker::PhantomData;

use crate::nn_modules::{MaxPooling, NNModuleBuilder, NNNecessaryTraits};

pub struct MaxPoolingBuilder<T> {
    _window_size: (usize, usize),
    _stride: usize,
    _phantom: PhantomData<T>,
}

impl<T> NNModuleBuilder<T> for MaxPoolingBuilder<T>
where
    T: NNNecessaryTraits,
{
    type BuiltObject = MaxPooling<T>;

    fn new() -> Self {
        let _window_size = (1usize, 1usize);
        let _stride = 1usize;

        MaxPoolingBuilder::<T> {
            _window_size,
            _stride,
            _phantom: PhantomData,
        }
    }

    fn build(self) -> Self::BuiltObject {
        let window_size = self._window_size;
        let stride = self._stride;
        let input_shape = Vec::<usize>::new();

        MaxPooling::<T> {
            window_size,
            stride,
            pooling_mask: None,
            input_shape,
            _phantom: PhantomData,
        }
    }
}

impl<T> MaxPoolingBuilder<T> {
    pub fn window_size(mut self, size: (usize, usize)) -> Self {
        self._window_size = size;
        self
    }

    pub fn stride(mut self, value: usize) -> Self {
        self._stride = value;
        self
    }
}
