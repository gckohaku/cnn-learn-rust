use ndarray::{ArrayD, ArrayViewD, Axis, Ix4, Zip};

use crate::{impl_as_any_with_mut, nn_modules::{NNForwardInput, NNModule, NNNecessaryTraits}};

#[derive(Debug, Clone)]
pub struct BatchNorm2d<T> {
	beta_average: T,
	gamma_variance: T,
}

impl<T> NNModule<T> for BatchNorm2d<T> where T: NNNecessaryTraits{
	fn necessary_parameter_value(&self) -> usize {
		1
	}
	
	fn forward(&mut self, input: &NNForwardInput<'_, '_, T>, is_grad: bool) -> Result<ArrayD<T>, &'static str> {
		let input_images = input.inputs[0];
		let input_images_4d = input_images.into_dimensionality::<Ix4>().unwrap();
		input_images_4d.permute_axes([0, 2, 3, 1]);
		let input_permute_shape = input_images_4d.shape();
		let input_images_2d =  input_images_4d.to_shape([input_permute_shape[0] * input_permute_shape[1] * input_permute_shape[2], input_permute_shape[3]]).unwrap();

		let mean_per_channel = input_images_2d.mean_axis(Axis(1)).unwrap();
		let variance_per_channel = input_images_2d.var_axis(Axis(1), T::ZERO);
		let inverse_std_per_channel = Zip::from(&variance_per_channel).par_map_collect(|var| T::ONE / T::sqrt(*var + T::from(1e-5).unwrap()));

		let norm_images_2d = inverse_std_per_channel * (input_images_2d - mean_per_channel);

		let result_2d = (norm_images_2d * self.gamma_variance) + self.beta_average;

		let mut result_4d = result_2d.to_shape([input_permute_shape[0], input_permute_shape[1], input_permute_shape[2], input_permute_shape[3]]).unwrap();
		result_4d.permute_axes([0, 3, 1, 2]);

		Ok(result_4d.to_owned().into_dyn())
	}

	fn propagate_grad(&mut self, grad: Option<&ArrayViewD<T>>, eta: T) -> ArrayD<T> {
		
	}

	impl_as_any_with_mut!();
}