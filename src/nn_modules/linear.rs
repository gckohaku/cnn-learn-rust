use ndarray::{Array1, Array2, ArrayViewD, Ix2};

use crate::nn_modules::NNModule;

/// アフィン変換を行うニューラルネットワークモジュール
pub struct Linear {
    pub weights: Array2<f64>,
    pub biases: Array1<f64>,
    pub is_grad: bool,
    // 重みを更新するために保持するデータ
    pub(super) input_value: Option<Array2<f64>>,
    pub(super) grad: Option<Array2<f64>>,
}

impl NNModule for Linear {
    type OutputArray = Array2<f64>;
    fn forward<'a>(&mut self, input: ArrayViewD<f64>) -> Array2<f64> {
        let input_2d = input.into_dimensionality::<Ix2>().unwrap();

        if self.is_grad == true {
            self.grad = Some(input_2d.t().to_owned());
        }

        input_2d.dot(&self.weights) + &self.biases
    }
}


