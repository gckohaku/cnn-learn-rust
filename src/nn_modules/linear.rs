use std::ops::Not;

use ndarray::{Array1, Array2, ArrayD, Ix2, LinalgScalar, ScalarOperand};

use crate::nn_modules::{HasOne, HasZero, NNForwardInput, NNModule};

/// アフィン変換を行うニューラルネットワークモジュール
pub struct Linear<T: Clone + Send + Sync> {
    pub weights: Array2<T>,
    pub biases: Array1<T>,
    pub is_grad: bool,
    // 重みを更新するために保持するデータ
    pub(super) input_value: Option<Array2<T>>,
    pub(super) grad: Option<Array2<T>>,
}

impl<T> NNModule<T> for Linear<T>
where
    T: ScalarOperand + Send + Sync + LinalgScalar,
{
    fn forward<'a>(&mut self, forward_input: NNForwardInput<'_, T>) -> ArrayD<T> {
        let input = forward_input.input;
        let input_2d = input.into_dimensionality::<Ix2>().unwrap();

        if self.is_grad == true {
            self.grad = Some(input_2d.t().to_owned());
        }

        let result = input_2d.dot(&self.weights) + &self.biases;
        result.into_dyn()
    }
}
