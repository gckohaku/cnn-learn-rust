pub mod linear;
pub mod relu;
pub mod builders;

pub trait NNModule {
	type InputArray;
	type OutputArray;
	fn forward(&self, input: Self::InputArray) -> Self::OutputArray;
}

pub trait NNModuleBuilder {
	type BuiltObject;
	fn new() -> Self;
	fn build(self) -> Self::BuiltObject;
}