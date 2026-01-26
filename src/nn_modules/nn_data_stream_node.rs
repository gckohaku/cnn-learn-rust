use std::ops::Deref;

use ndarray::ArrayD;

use crate::nn_modules::{NNForwardInput, NNModule};

// ループごとに新しくインスタンスを生成するか、Vec<Arc<T>> にすることを検討した方がいいかも
pub struct NNDataStreamNode<'a, T> {
    pub module_data: &'a mut dyn NNModule<T>,
    pub children: Vec<NNDataStreamNode<'a, T>>,
}

impl<'a, T> NNModule<T> for NNDataStreamNode<'a, T>
where
    T: Clone,
{
    fn forward(&mut self, input: &NNForwardInput<'_, '_, T>) -> ndarray::ArrayD<T> {
        let data = &self.module_data.forward(input);
        data.clone()
    }
}

impl<'a, T> NNDataStreamNode<'a, T> {
    pub fn add_child<'b>(&mut self, child_module: &'b mut dyn NNModule<T>) -> NNDataStreamNode<'b, T> {
        NNDataStreamNode::<'b, T> {
            module_data: child_module,
            children: Vec::<NNDataStreamNode<'_, T>>::new(),
        }
    }
}
