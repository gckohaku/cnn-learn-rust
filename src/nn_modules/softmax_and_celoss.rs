use ndarray::Array2;

use crate::nn_modules::{CrossEntropyLoss, NNModule, Softmax, softmax};

pub struct SoftmaxAndCELoss {
    pub softmax: Softmax,
    pub cross_entropy_loss: CrossEntropyLoss,
    pub is_grad: bool,
    // 勾配を求める時に利用
    pub(super) grad: Option<Array2<f64>>,
}

impl NNModule for SoftmaxAndCELoss {
    type InputArray<'a> = (&'a Array2<f64>, &'a Array2<f64>);
    type OutputArray = f64;

    fn forward<'a>(&mut self, input: (&'a Array2<f64>, &'a Array2<f64>)) -> f64 {
        let input_values = input.0;
        let target_values = input.1;

		let softmax_result = self.softmax.forward(input_values);
        let loss = self.cross_entropy_loss.forward((&softmax_result, target_values));
		self.grad = Some(softmax_result - target_values);
		loss
    }
}
