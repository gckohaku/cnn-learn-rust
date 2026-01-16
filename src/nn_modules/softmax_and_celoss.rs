use ndarray::{Array2, ArrayD, ArrayViewD};

use crate::nn_modules::{CrossEntropyLoss, NNForwardInput, NNModule, Softmax, softmax};

pub struct SoftmaxAndCELoss {
    pub softmax: Softmax,
    pub cross_entropy_loss: CrossEntropyLoss,
    pub target: Option<Array2<f64>>,
    pub is_grad: bool,
    // 勾配を求める時に利用
    pub(super) grad: Option<Array2<f64>>,
}

impl<T> NNModule for SoftmaxAndCELoss {
    fn forward<'a>(&mut self, input: NNForwardInput<'_, T> ) -> ArrayD<f64> {
        let input_values = input;
        let target_values = self.target.unwrap();

		let softmax_result = self.softmax.forward(input_values);
        let loss = self.cross_entropy_loss.forward(softmax_result);
		self.grad = Some(softmax_result - target_values);
		loss
    }
}
