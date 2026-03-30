use std::marker::PhantomData;

use ndarray::{Array1, Array4};

use crate::{
    nn_modules::{Convolution, NNModuleBuilder, NNModuleType, NNNecessaryTraits},
    rand::Rand,
};

pub struct ConvolutionBuilder<T> {
    _input_channel_value: usize,
    _input_image_size: (usize, usize),
    _filter_value: usize,
    _filter_size: (usize, usize),
    _stride: usize,
    _padding: usize,
    _phantom: PhantomData<T>,
}

impl<T> NNModuleBuilder<T> for ConvolutionBuilder<T>
where
    T: NNNecessaryTraits,
{
    type BuiltObject = NNModuleType<T>;

    fn new() -> Self {
        let _input_channel_value = 1usize;
        let _input_image_size = (1usize, 1usize);
        let _filter_value = 1usize;
        let _filter_size = (1usize, 1usize);
        let _stride = 1usize;
        let _padding = 0usize;

        Self {
            _input_channel_value,
            _input_image_size,
            _filter_value,
            _filter_size,
            _stride,
            _padding,
            _phantom: PhantomData,
        }
    }

    fn build(self) -> Self::BuiltObject {
        let mut filters = Array4::<T>::zeros((
            self._filter_value,
            self._input_channel_value,
            self._filter_size.0,
            self._filter_size.1,
        ));
        let biases = Array1::<T>::zeros(self._filter_value);
        let stride = self._stride;
        let padding = self._padding;
        let filter_size = self._filter_size;

        // 重みの He 初期化
        let mut r = Rand::new();

        filters.mapv_inplace(|_x| {
            r.normal(
                T::ZERO,
                T::from(
                    2.0 / (self._input_channel_value
                        * self._input_image_size.0
                        * self._input_image_size.1) as f64,
                )
                .expect("input size cannot to cast from usize to T")
                .sqrt(),
            )
        });

        NNModuleType::Convolution(Convolution::<T> {
            filters,
            biases,
            stride,
            padding,
            filter_size,
            input_of_forward: None,
        })
    }
}

impl<T> ConvolutionBuilder<T> {
    pub fn input_channel_value(mut self, value: usize) -> Self {
        self._input_channel_value = value;
        self
    }

    pub fn input_image_size(mut self, size: (usize, usize)) -> Self {
        self._input_image_size = size;
        self
    }

    pub fn filter_value(mut self, value: usize) -> Self {
        self._filter_value = value;
        self
    }

    pub fn filter_size(mut self, size: (usize, usize)) -> Self {
        self._filter_size = size;
        self
    }

    pub fn stride(mut self, value: usize) -> Self {
        self._stride = value;
        self
    }

    pub fn padding(mut self, value: usize) -> Self {
        self._padding = value;
        self
    }
}
