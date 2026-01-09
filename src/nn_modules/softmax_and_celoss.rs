use ndarray::Array2;

use crate::nn_modules::{CrossEntropyLoss, Softmax};

pub struct SoftmaxAndCELoss {
	pub softmax: Softmax,
	pub cross_entropy_loss: CrossEntropyLoss,
	pub is_grad: bool,
	// 勾配を求める時に利用
	pub(super) grad: Option<Array2<f64>>,
}