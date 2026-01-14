pub mod builders;
pub mod linear;
pub mod relu;
pub mod softmax;
pub mod cross_entropy_loss;
pub mod softmax_and_celoss;
pub mod nn_directed_graph;

pub use linear::Linear;
pub use relu::ReLU;
pub use softmax::Softmax;
pub use cross_entropy_loss::CrossEntropyLoss;
pub use softmax_and_celoss::SoftmaxAndCELoss;
pub use nn_directed_graph::NNDirectedGraph;

pub trait NNModule {
	type InputArray<'a>;
	type OutputArray;
	fn forward<'a>(&mut self, input: Self::InputArray<'a>) -> Self::OutputArray;
}

pub trait NNModuleBuilder {
	type BuiltObject;
	fn new() -> Self;
	fn build(self) -> Self::BuiltObject;
}