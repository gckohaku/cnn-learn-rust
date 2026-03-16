use ndarray::{Array2, array};

use crate::nn_modules::{
    NNDataFlowTree, NNForwardInput, NNModule, NNModuleBuilder, builders::{LinearBuilder, ReLUBuilder, SoftmaxAndCELossBuilder}
};

type ElementType = f32;

pub fn run_test() {
    let input_size = 28 * 28;

    let mut tree = NNDataFlowTree::<ElementType>::new();

    let mut linear1 = LinearBuilder::<ElementType>::new()
        .input_node_value(4)
        .output_node_value(3)
        .is_grad(true)
        .build();
    let relu = ReLUBuilder::<ElementType>::new().is_grad(true).build();
    let mut linear2 = LinearBuilder::<ElementType>::new()
        .input_node_value(3)
        .output_node_value(3)
        .is_grad(true)
        .build();
    let output_module = SoftmaxAndCELossBuilder::<ElementType>::new().is_grad(true).build();

    #[cfg(debug_assertions)]
    {
        use crate::nn_modules::{NNModuleType};
        match linear1 {
            NNModuleType::Linear(ref mut l) => {
                use ndarray::array;

                l.debug_set_weights(array![
                    [0.6, 0.2, -0.5],
                    [-0.2, 0.0, -0.8],
                    [0.9, 0.3, 0.1],
                    [0.4, -0.8, -0.3],
                ]);
                l.debug_set_biases(array![0.0, 0.1, 0.2]);
            }
            _ => panic!("Unexpected Behavior"),
        }

        match linear2 {
            NNModuleType::Linear(ref mut l) => {
                use ndarray::array;

                l.debug_set_weights(array![
                    [0.3, 0.2, -0.5],
                    [-0.2, 0.8, 0.7],
                    [0.6, -0.4, -0.1],
                ]);
                l.debug_set_biases(array![0.2, -0.1, 0.1]);
            }
            _ => panic!("Unexpected Behavior"),
        }
    }

    let linear_info = tree.add_from_root(linear1);
    let relu_info = tree.add(&linear_info, relu);
    let linear2_info = tree.add(&relu_info, linear2);
    let output_info = tree.add(&linear2_info, output_module);

    // これは初回 forward 時に自動実行するようにしたい
    tree.calc_sequential_node_flow();

    dbg!(&tree);

	let input = Array2::<ElementType>::from(array![[0.5, 0.1, 1.0, 0.25]]).into_dyn();
	let target = Array2::<ElementType>::from(array![[1.0, 0.0, 0.0]]).into_dyn();
	let error = &tree.forward(&NNForwardInput::<ElementType> {
		inputs: vec![input.view()],
		target: Some(target.view()),
	});

	println!("error: {}", error);

    tree.propagate_grad(None, 0.1);
}
