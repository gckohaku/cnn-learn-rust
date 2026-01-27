use ndarray::{
    ArrayD, IxDyn,
};
use num_traits::Zero;

use crate::nn_modules::{NNDataStreamNode, NNForwardInput, NNModule};

pub struct NNDataFlowTree<'a, T> {
    pub tree: Vec<NNDataStreamNode<'a, T>>,
	result: ArrayD<T>,
}

struct TopologicalSortedList<'a, T> {
	
}

impl<'a, T> NNModule<T> for NNDataFlowTree<'a, T>
where
    T: Clone + Zero,
{
    fn forward(&mut self, input: &NNForwardInput<'_, '_, T>) -> ArrayD<T> {
        for node in &self.tree {}

        ArrayD::<T>::zeros(IxDyn(&vec![0]))
    }
}

fn recursive_forward<'a, T>(stream_node: &mut NNDataStreamNode<'a, T>, input: &NNForwardInput<'_, '_, T>)
where
    T: Clone + Zero,
{
	let data = stream_node.module_data.forward(input);

	if stream_node.children.len() == 0 {
		
	}

	for node in &stream_node.children {
		_ = recursive_forward(stream_node, input);
	}
}

impl<'a, T> NNDataFlowTree<'a, T> {
    pub fn new() -> Self {
        NNDataFlowTree::<T> {
            tree: Vec::<NNDataStreamNode<'_, T>>::new(),
        }
    }
}
