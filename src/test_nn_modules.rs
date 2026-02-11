use crate::nn_modules::{NNDataFlowTree, NNModuleBuilder, builders::{LinearBuilder, ReLUBuilder}, linear};

type ElementType = f32;

pub fn run_test() {
	let mut tree = NNDataFlowTree::<ElementType>::new();

	let linear1 = LinearBuilder::<ElementType>::new().input_node_value(10).output_node_value(20).build();
	let relu = ReLUBuilder::<ElementType>::new().build();

	let linear_info = tree.add_from_root(&linear1);
	let relu_info = tree.add(&linear_info, &relu);

	dbg!(tree);
}