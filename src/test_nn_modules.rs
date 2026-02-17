use crate::nn_modules::{NNDataFlowTree, NNModuleBuilder, builders::{LinearBuilder, ReLUBuilder, SoftmaxAndCELossBuilder}};

type ElementType = f32;

pub fn run_test() {
	let input_size = 28 * 28;

	let mut tree = NNDataFlowTree::<ElementType>::new();

	let linear1 = LinearBuilder::<ElementType>::new().input_node_value(input_size).output_node_value(100).build();
	let relu = ReLUBuilder::<ElementType>::new().build();
	let linear2 = LinearBuilder::<ElementType>::new().input_node_value(100).output_node_value(10).build();
	let output_module = SoftmaxAndCELossBuilder::<ElementType>::new().build();

	let linear_info = tree.add_from_root(&linear1);
	let relu_info = tree.add(&linear_info, &relu);
	let linear2_info = tree.add(&relu_info, &linear2);
	let output_info = tree.add(&linear2_info, &output_module);

	dbg!(tree);
}