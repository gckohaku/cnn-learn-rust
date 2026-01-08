pub mod linear;
pub mod relu;
pub mod builders;
pub mod softmax;
pub mod cross_entropy_loss;

pub use linear::Linear;
pub use relu::ReLU;
pub use softmax::Softmax;
pub use cross_entropy_loss::CrossEntropyLoss;

pub trait NNModule {
	type InputArray;
	type OutputArray;
	fn forward(&mut self, input: Self::InputArray) -> Self::OutputArray;
}

pub trait NNModuleBuilder {
	type BuiltObject;
	fn new() -> Self;
	fn build(self) -> Self::BuiltObject;
}