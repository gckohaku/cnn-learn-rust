use std::marker::PhantomData;

use ndarray::{Array1, Array2};
use num_traits::Zero;

use crate::nn_modules::{NNModuleBuilder, linear::Linear};


pub struct LinearBuilder<T> {
    _input_node_value: usize,
    _output_node_value: usize,
    _is_grad: bool,
    _phantom: PhantomData<T>,
}

impl<T> NNModuleBuilder<T> for LinearBuilder<T>
where
    T: Clone + Zero,
{
    type BuiltObject = Linear<T>;

    fn new() -> Self {
        let input_node_value = 0;
        let output_node_value = 0;
        let is_grad = false;

        LinearBuilder::<T> {
            _input_node_value: input_node_value,
            _output_node_value: output_node_value,
            _is_grad: is_grad,
            _phantom: PhantomData,
        }
    }

    fn build(self) -> Linear<T> {
        let weights = Array2::<T>::zeros((self._input_node_value, self._output_node_value));
        let biases = Array1::<T>::zeros(self._input_node_value);

        Linear::<T> {
            weights,
            biases,
            is_grad: self._is_grad,
            input_value: None,
            grad: None,
        }
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

    pub fn is_grad(mut self, value: bool) -> Self {
        self._is_grad = value;
        self
    }
}
