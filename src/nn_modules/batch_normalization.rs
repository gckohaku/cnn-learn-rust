use ndarray::{ArrayD, ArrayViewD, Axis};

use crate::{impl_as_any_with_mut, nn_modules::{NNForwardInput, NNModule, NNNecessaryTraits}};

pub struct BatchNormalization<T> {
	beta_average: T,
	gamma_variance: T,
}

impl<T> NNModule<T> for BatchNormalization<T> where T: NNNecessaryTraits{
	fn necessary_parameter_value(&self) -> usize {
		1
	}
	
	fn forward(&mut self, input: &NNForwardInput<'_, '_, T>, is_grad: bool) -> Result<ArrayD<T>, &'static str> {
		let input_images = input.inputs[0];
		let mean = input_images.mean_axis(Axis(0));
		let variance = input_images.var_axis(Axis(0), T::ZERO);
	}

	fn propagate_grad(&mut self, grad: Option<&ArrayViewD<T>>, eta: T) -> ArrayD<T> {
		
	}

	impl_as_any_with_mut!();
}