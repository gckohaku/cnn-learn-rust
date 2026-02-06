use std::{cell::RefCell, rc::Rc};

use crate::nn_modules::{NNDataFlowTree, NNModule};

pub struct NNDataFlowNode<T> {
	pub index: usize,
	pub add_module_callback: Box<dyn FnMut(usize, Box<dyn NNModule<T>>) -> NNDataFlowNode<T>>,
}