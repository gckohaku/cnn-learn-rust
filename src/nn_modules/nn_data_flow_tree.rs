use crate::nn_modules::NNModule;

pub struct NNDataFlowTree<'a, T> {
	pub tree: Vec<NNDataFlowNode<'a, T>>,
}

pub struct NNDataFlowNode<'a, T> {
	pub data: &'a dyn NNModule<T>,
	pub children: Vec<NNDataFlowNode<'a, T>>,
}

impl<'a, T> NNDataFlowTree<'a, T> {
	pub fn new() -> Self {
		NNDataFlowTree::<T> {
			tree: Vec::<NNDataFlowNode<'_, T>>::new(),
		}
	}
}