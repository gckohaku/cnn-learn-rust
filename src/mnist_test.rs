use mnist::{MnistBuilder, NormalizedMnist};
use ndarray::{Array2, Array4};
use std::{io::Write, time};

use crate::{
    cnn_information::{
        ActivationType, ConvolutionInformation, LayerInformation, LayerInformationContent,
        LayerType, OutputInformation, OutputType, PoolingInformation, PoolingType,
    }, cnn_network::NeuralNetworkCNN, nn_modules::{NNDataFlowTree, NNModuleBuilder, builders::{LinearBuilder, ReLUBuilder, SoftmaxAndCELossBuilder}}, rand::Rand, utilities::shuffle
};

const IMAGE_ROW_SIZE: usize = 28;
const IMAGE_DOT_VALUE: usize = IMAGE_ROW_SIZE * IMAGE_ROW_SIZE;
const IMAGE_CHANNEL_VALUE: usize = 1;

pub fn mnist_process() {
    let epoch_value = 10;
    let mini_batch_sample_size = 1000;

    let training_value = 10000;
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
    let mut linear1 = LinearBuilder::new().input_node_value(784).output_node_value(196).is_grad(true).build();
    let mut relu1 = ReLUBuilder::new().is_grad(true).build();
    let mut linear2 = LinearBuilder::new().input_node_value(196).output_node_value(49).is_grad(true).build();
    let mut relu2 = ReLUBuilder::new().is_grad(true).build();
    let mut linear3 = LinearBuilder::new().input_node_value(49).output_node_value(10).is_grad(true).build();
    let mut softmax_and_celoss = SoftmaxAndCELossBuilder::new().is_grad(true).build();

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

        for indices in shuffle_index.chunks(mini_batch_sample_size) {
            mini_batch_count += 1;
            let (inputs, expects) = make_mini_batch_dataset(indices, &mnist);

            nn.forward(&inputs, &expects);
            epoch_error += nn.get_error();
            nn.backward(&expects, 0.01);

            print!("\rmini batch count: {}", mini_batch_count);
            std::io::stdout().flush().unwrap();
        }

        println!(
            "\nepoch {:6} error: {:13.10}",
            epoch,
            epoch_error / training_value as f64
        );
        dbg!(nn.get_output());

    }

    // 処理時間表示
    println!(
        "epochs process duration: {:?}sec.",
        epochs_now.elapsed().as_secs_f64()
    );
}

fn make_mini_batch_dataset(
    indices: &[usize],
    dataset: &NormalizedMnist,
) -> (Array4<f64>, Array2<f64>) {
    let sample_size = indices.len();

    let mut inputs_array = Vec::<f64>::new();
    let mut expected_array = Vec::<f64>::new();

    for index in indices {
        let trn_data: &Vec<f64> = &dataset.trn_img
            [((*index) * IMAGE_DOT_VALUE)..(((*index) + 1) * IMAGE_DOT_VALUE)]
            .iter()
            .map(|&x| x as f64)
            .collect();
        inputs_array.extend_from_slice(trn_data);

        let trn_label: &Vec<f64> = &dataset.trn_lbl[((*index) * 10)..(((*index) + 1) * 10)]
            .iter()
            .map(|&x| x as f64)
            .collect();
        expected_array.extend_from_slice(trn_label);
    }

    let inputs = Array4::<f64>::from_shape_vec(
        (
            sample_size,
            IMAGE_CHANNEL_VALUE,
            IMAGE_ROW_SIZE,
            IMAGE_ROW_SIZE,
        ),
        inputs_array,
    )
    .unwrap();
    let expects = Array2::<f64>::from_shape_vec((sample_size, 10), expected_array).unwrap();

    (inputs, expects)
}
