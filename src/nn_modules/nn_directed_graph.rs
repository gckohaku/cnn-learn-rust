use ndarray::ArrayD;

use crate::nn_modules::{NNForwardInput, NNGraphNode, NNModule};

pub struct NNDirectedGraph<'a, T> {
	pub graph: NNGraphNode<'a, T>
}

impl<'a, T> NNModule<T> for NNDirectedGraph<'a, T> {
	fn forward(&mut self, input: &NNForwardInput<'_, '_, T>) -> ArrayD<T> {
		let parent_data = self.graph.module_data.forward(input);
	}
}

impl<'a, T> NNDirectedGraph<'a, T> {
	pub fn recursion_forward(&mut self, input: &NNForwardInput<'_, '_, T>, node: &NNGraphNode<'a, T>) -> ArrayD<T> {
		let parent_data = self.graph.module_data.forward(input);
	}
}