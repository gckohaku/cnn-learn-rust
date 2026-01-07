use ndarray::ArrayD;

use crate::nn_modules::NNModule;

pub struct ReLU {
	pub inplace: bool,
	// 勾配計算のために保持するデータ
	pub(super) output_value: Option<ArrayD<f64>>,
}

impl NNModule for ReLU {
	type InputArray = ArrayD<f64>;
	type OutputArray = ArrayD<f64>;

	fn forward(&mut self, mut input: Self::InputArray) -> Self::OutputArray {
		if self.inplace {
			input.par_mapv_inplace(|x| x.max(0.0));
			self.output_value = Some(input.clone());
			return input;
		}
		else {
			let mut clone_array = input.clone();
			clone_array.par_mapv_inplace(|x| x.max(0.0));
			self.output_value = Some(input.clone());
			return clone_array;
		}
	}
}