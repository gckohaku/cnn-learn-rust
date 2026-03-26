use std::marker::PhantomData;

use crate::nn_modules::{NNModule, NNNecessaryTraits};

#[derive(Debug)]
pub struct Flatten<T> {
    _shape: Vec<usize>,
    _before_shape: Option<Vec<usize>>,
    _phantom: PhantomData<T>,
}

impl<T> NNModule<T> for Flatten<T>
where
    T: NNNecessaryTraits,
{
    fn necessary_parameter_value(&self) -> usize {
        1
    }

    fn forward(
        &mut self,
        input: &super::NNForwardInput<'_, '_, T>,
        is_grad: bool,
    ) -> ndarray::ArrayD<T> {
        let input_tensor = &input.inputs[0];

		if is_grad {
			let shape = input_tensor.shape();
			self._before_shape = Some(shape.iter().cloned().collect());
		}
        

        let shaped_tensor = input_tensor.to_shape(self._shape.clone()).unwrap();
		shaped_tensor.to_owned()
    }

	fn propagate_grad(&mut self, grad: Option<&ndarray::ArrayViewD<T>>, _eta: T) -> ndarray::ArrayD<T> {
		let propagated_tensor = grad.unwrap();
		let shaped_tensor = propagated_tensor.to_shape(self._before_shape.to_owned().unwrap());
		shaped_tensor.unwrap().to_owned()
	}
}
