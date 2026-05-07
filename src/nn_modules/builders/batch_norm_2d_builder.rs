use std::marker::PhantomData;

use ndarray::{Array1, Array2};

use crate::nn_modules::{BatchNorm2d, NNModuleBuilder, NNNecessaryTraits};

pub struct BatchNorm2dBuilder<T> {
    _channel_size: usize,
    _phantom: PhantomData<T>,
}

impl<T> NNModuleBuilder<T> for BatchNorm2dBuilder<T>
where
    T: NNNecessaryTraits,
{
    type BuiltObject = BatchNorm2d<T>;

    fn new() -> Self {
        let _channel_size = 1usize;

        Self {
            _channel_size,
            _phantom: PhantomData,
        }
    }

    fn build(self) -> Self::BuiltObject {
        let beta_average = Array1::<T>::zeros(self._channel_size);
        let gamma_variance = Array1::<T>::ones(self._channel_size);

        let std_input = Array2::<T>::zeros((1, self._channel_size));
        let input_minus_mu = Array2::<T>::zeros((1, self._channel_size));
        let input_variance = Array1::<T>::zeros(self._channel_size);

        let running_average = Array1::<T>::zeros(self._channel_size);
        let running_variance = Array1::<T>::zeros(self._channel_size);

        BatchNorm2d::<T> {
            beta_average,
            gamma_variance,
            std_input,
            input_minus_mu,
            input_variance,
            running_average,
            running_variance,
        }
    }
}

impl<T> BatchNorm2dBuilder<T> {
	pub fn channel_size(mut self, size: usize) -> Self {
		self._channel_size = size;
		self
	}
}