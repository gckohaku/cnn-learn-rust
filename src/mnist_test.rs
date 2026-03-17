use mnist::{MnistBuilder, NormalizedMnist};
use ndarray::{Array2, Ix0};
use std::{io::Write, time};

use crate::{
    nn_modules::{
        NNDataFlowTree, NNForwardInput, NNModule, NNModuleBuilder,
        builders::{LinearBuilder, ReLUBuilder, SoftmaxAndCELossBuilder},
    },
    rand::Rand,
    utilities::shuffle,
};

type ElementType = f32;

const IMAGE_ROW_SIZE: usize = 28;
const IMAGE_DOT_VALUE: usize = IMAGE_ROW_SIZE * IMAGE_ROW_SIZE;
const IMAGE_CHANNEL_VALUE: usize = 1;

pub fn mnist_process() {
    let epoch_value = 10;
    let mini_batch_sample_size = 500;

    let training_value = 60000;
    let validation_value = 5000;
    let test_value = 5000;

    let mnist = MnistBuilder::new()
        .label_format_one_hot()
        .training_set_length(training_value)
        .validation_set_length(validation_value)
        .test_set_length(test_value)
        .training_images_filename("train-images.idx3-ubyte")
        .training_labels_filename("train-labels.idx1-ubyte")
        .test_images_filename("t10k-images.idx3-ubyte")
        .test_labels_filename("t10k-labels.idx1-ubyte")
        .finalize()
        .normalize();

    let mut r = Rand::new();

    // ニューラルネットワークの作成
    let mut tree = NNDataFlowTree::new();
    let linear1 = LinearBuilder::new()
        .input_node_value(784)
        .output_node_value(196)
        .build();
    let relu1 = ReLUBuilder::new().build();
    let linear2 = LinearBuilder::new()
        .input_node_value(196)
        .output_node_value(49)
        .build();
    let relu2 = ReLUBuilder::new().build();
    let linear3 = LinearBuilder::new()
        .input_node_value(49)
        .output_node_value(10)
        .build();
    let softmax_and_celoss = SoftmaxAndCELossBuilder::new().build();

    let linear_info1 = &tree.add_from_root(linear1);
    let relu_info1 = &tree.add(&linear_info1, relu1);
    let linear_info2 = &tree.add(&relu_info1, linear2);
    let relu_info2 = &tree.add(&linear_info2, relu2);
    let linear_info3 = &tree.add(&relu_info2, linear3);
    _ = &tree.add(&linear_info3, softmax_and_celoss);

    // 処理時間計測用
    let epochs_now = time::Instant::now();

    for epoch in 1..=epoch_value {
        let shuffle_index = shuffle::generate_shuffle_array(training_value as usize, &mut r);
        let mut epoch_error = 0.0;

        let mut mini_batch_count = 0;

        for indices in shuffle_index.chunks(mini_batch_sample_size as usize) {
            mini_batch_count += 1;
            let (inputs, expects) = make_mini_batch_dataset(indices, &mnist);

            epoch_error += &tree
                .forward(
                    &NNForwardInput::<ElementType> {
                        inputs: vec![inputs.view().into_dyn()],
                        target: Some(expects.view().into_dyn()),
                    },
                    true,
                )
                .into_dimensionality::<Ix0>()
                .unwrap().into_scalar();
            _ = &tree.propagate_grad(None, 0.001);

            print!("\rmini batch count: {}", mini_batch_count);
            std::io::stdout().flush().unwrap();
        }

        println!(
            "\nepoch {:6} error: {:13.10}",
            epoch,
            epoch_error / training_value as ElementType
        );
    }

    // 処理時間表示
    println!(
        "epochs process duration: {:?}sec.",
        epochs_now.elapsed().as_secs_f64()
    );
}

fn make_mini_batch_dataset<'a>(
    indices: &'a [usize],
    dataset: &'a NormalizedMnist,
) -> (Array2<ElementType>, Array2<ElementType>) {
    let sample_size = indices.len();

    let mut inputs_array = Vec::<ElementType>::new();
    let mut expected_array = Vec::<ElementType>::new();

    for index in indices {
        let trn_data: &Vec<ElementType> = &dataset.trn_img
            [((*index) * IMAGE_DOT_VALUE)..(((*index) + 1) * IMAGE_DOT_VALUE)]
            .iter()
            .map(|&x| x as ElementType)
            .collect();
        inputs_array.extend_from_slice(trn_data);

        let trn_label: &Vec<ElementType> = &dataset.trn_lbl[((*index) * 10)..(((*index) + 1) * 10)]
            .iter()
            .map(|&x| x as ElementType)
            .collect();
        expected_array.extend_from_slice(trn_label);
    }

    let inputs = Array2::<ElementType>::from_shape_vec(
        (
            sample_size,
            IMAGE_CHANNEL_VALUE * IMAGE_DOT_VALUE,
        ),
        inputs_array,
    )
    .unwrap();
    let expects = Array2::<ElementType>::from_shape_vec((sample_size, 10), expected_array).unwrap();

    (inputs.to_owned(), expects.to_owned())
}
