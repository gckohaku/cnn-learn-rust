use ndarray::{Array1, Array2};

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
    type InputArray<'a> = &'a Array2<f64>;
    type OutputArray = Array2<f64>;
    fn forward<'a>(&mut self, input: &'a Array2<f64>) -> Array2<f64> {
        if self.is_grad == true {
            self.grad = Some(input.clone().t().to_owned());
        }

        input.dot(&self.weights) + &self.biases
    }
}


