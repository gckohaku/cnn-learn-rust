use std::{cell::RefCell, rc::Rc};

use crate::nn_modules::{NNDataFlowTree, NNModule};

#[derive(Debug)]
pub struct NNDataFlowNode<'a, T> {
	pub index: usize,
	pub tree_module: &'a mut NNDataFlowTree<'a, T>,
}


impl<'a, T> NNDataFlowNode<'a, T> {
	pub fn add_module(&'a mut self, module: &'a dyn NNModule<T>) -> NNDataFlowNode<'_, T> {
		self.tree_module.add_module_for_node(self.index, module)
	}
}