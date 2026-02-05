use std::rc::Rc;

use ndarray::{ArrayD, IxDyn};
use num_traits::Zero;

use crate::nn_modules::{NNDataFlowNode, NNForwardInput, NNModule};

pub struct NNDataFlowTree<T> {
    pub modules: Vec<Box<dyn NNModule<T>>>,
    current_count: usize,
}

// impl<'a, T> NNModule<T> for NNDataFlowTree<T>
// where
//     T: Clone + Zero,
// {
//     fn forward(&mut self, input: &NNForwardInput<'_, '_, T>) -> ArrayD<T> {

//     }
// }

impl<T> NNDataFlowTree<T> {
    pub fn add_module(&mut self, index: usize, module: Box<dyn NNModule<T>>) -> NNDataFlowNode<T> {
        self.modules.push(module);
        self.current_count += 1;
        NNDataFlowNode::<T> {
            index: index,
            tree: &self.add_module,
        }
    }
}
