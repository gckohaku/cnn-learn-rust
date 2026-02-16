use ndarray::ArrayD;

use crate::nn_modules::NNModule;

#[derive(Debug)]
pub struct InputTensor<T> {
    pub data: ArrayD<T>,
}

impl<T> NNModule<T> for InputTensor<T>
where
    T: Clone + std::fmt::Debug,
{
    fn forward(&mut self, input: &super::NNForwardInput<'_, '_, T>) -> ArrayD<T> {
        return self.data.to_owned();
    }
}
