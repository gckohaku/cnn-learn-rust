use std::marker::PhantomData;

use ndarray::{Array1, Array2};

use crate::{
    nn_modules::{NNModuleBuilder, NNModuleType, NNNecessaryTraits, linear::Linear},
    rand::Rand,
};

pub struct LinearBuilder<T> {
    _input_node_value: usize,
    _output_node_value: usize,
    _phantom: PhantomData<T>,
}

impl<T> NNModuleBuilder<T> for LinearBuilder<T>
where
    T: NNNecessaryTraits,
{
    type BuiltObject = NNModuleType<T>;

    fn new() -> Self {
        let input_node_value = 0;
        let output_node_value = 0;

        LinearBuilder::<T> {
            _input_node_value: input_node_value,
            _output_node_value: output_node_value,
            _phantom: PhantomData,
        }
    }

    fn build(self) -> NNModuleType<T> {
        let mut weights = Array2::<T>::zeros((self._input_node_value, self._output_node_value));
        let biases = Array1::<T>::zeros(self._output_node_value);

        // 重みの He 初期化
        let mut r = Rand::new();

        weights.mapv_inplace(|_x| {
            r.normal(
                T::ZERO,
                T::from(2.0 / self._input_node_value as f64)
                    .expect("input size cannot to cast from usize to T")
                    .sqrt(),
            )
        });

        NNModuleType::Linear(Linear::<T> {
            weights,
            biases,
            input_value: None,
            grad: None,
        })
    }
}

impl<T> LinearBuilder<T> {
    pub fn input_node_value(mut self, value: usize) -> Self {
        self._input_node_value = value;
        self
    }

    pub fn output_node_value(mut self, value: usize) -> Self {
        self._output_node_value = value;
        self
    }
}
