use std::fmt::Debug;

use ndarray::{Array1, Array2, ArrayD, Ix2, LinalgScalar};

use crate::nn_modules::{NNForwardInput, NNModule};

/// アフィン変換を行うニューラルネットワークモジュール
#[derive(Debug)]
pub struct Linear<T> {
    pub weights: Array2<T>,
    pub biases: Array1<T>,
    pub is_grad: bool,
    // 重みを更新するために保持するデータ
    pub(super) input_value: Option<Array2<T>>,
    pub(super) grad: Option<Array2<T>>,
}

impl<T> NNModule<T> for Linear<T>
where
    T: Send + Sync + LinalgScalar + Debug,
{
    fn forward<'a>(&mut self, forward_input: &NNForwardInput<'_, '_, T>) -> ArrayD<T> {
        let input = &forward_input.input;
        let input_2d = input.clone().into_dimensionality::<Ix2>().unwrap();

        if self.is_grad == true {
            self.grad = Some(input_2d.t().to_owned());
        }

        let result = input_2d.dot(&self.weights) + &self.biases;
        result.into_dyn()
    }
}
