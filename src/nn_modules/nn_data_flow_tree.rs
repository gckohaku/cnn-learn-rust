use crate::nn_modules::{NNDataStreamNode, NNModule};

pub struct NNDataFlowTree<'a, T> {
	pub tree: Vec<NNDataStreamNode<'a, T>>,
}

// impl<'a, T> NNModule<T> for NNDataFlowTree<'a, T> {
	
// }

impl<'a, T> NNDataFlowTree<'a, T> {
	pub fn new() -> Self {
		NNDataFlowTree::<T> {
			tree: Vec::<NNDataStreamNode<'_, T>>::new(),
		}
	}
}