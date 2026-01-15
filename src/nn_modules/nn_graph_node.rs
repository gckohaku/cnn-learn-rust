use crate::nn_modules::NNModule;

pub struct NNGraphNode<'a, Module: NNModule> {
	pub module_data: &'a Module,
	// pub child_nodes: Vec<&'a dyn NNModule>
}