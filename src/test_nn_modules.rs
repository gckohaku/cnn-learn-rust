use crate::nn_modules::{NNDataFlowTree, NNModuleBuilder, builders::{LinearBuilder, ReLUBuilder}};

type ElementType = f32;

pub fn run_test() {
	let mut tree = NNDataFlowTree::<ElementType>::new();

	let linear1 = LinearBuilder::<ElementType>::new().input_node_value(10).output_node_value(20).build();
	let relu = ReLUBuilder::<ElementType>::new().build();

	let mut linear1_node = tree.add_module(&linear1);
	let mut relu_node = linear1_node.add_module(&relu);
}