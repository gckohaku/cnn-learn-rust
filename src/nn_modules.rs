pub mod builders;
pub mod linear;
pub mod relu;
pub mod softmax;
pub mod cross_entropy_loss;
pub mod softmax_and_celoss;
pub mod nn_directed_graph;
pub mod nn_graph_node;
pub mod trait_implements;

pub use linear::Linear;
use ndarray::{ArrayD, ArrayViewD};
pub use relu::ReLU;
pub use softmax::Softmax;
pub use cross_entropy_loss::CrossEntropyLoss;
pub use softmax_and_celoss::SoftmaxAndCELoss;
pub use nn_directed_graph::NNDirectedGraph;
pub use nn_graph_node::NNGraphNode;

pub struct NNForwardInput<'a, T> {
	pub input: ArrayViewD<'a, T>,
	pub target: Option<ArrayViewD<'a, T>>,
}

pub trait NNModule<T> {
	fn forward(&mut self, input: NNForwardInput<'_, T>) -> ArrayD<f64>;
}

pub trait NNModuleBuilder {
	type BuiltObject;
	fn new() -> Self;
	fn build(self) -> Self::BuiltObject;
}