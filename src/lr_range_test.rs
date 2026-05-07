use mnist::{MnistBuilder, NormalizedMnist};
use ndarray::{Array2, Array4, Ix0};
use std::{
    collections::HashMap,
    fmt::format,
    fs::File,
    io::{self, BufRead, BufReader, Write},
    time,
};

use crate::{
    nn_modules::{
        NNDataFlowTree, NNForwardInput, NNModule, NNModuleBuilder, ReshapeTensor, SoftmaxAndCELoss,
        builders::{
            BatchNorm2dBuilder, ConvolutionBuilder, LinearBuilder, MaxPoolingBuilder, ReLUBuilder, ReshapeTensorBuilder, SoftmaxAndCELossBuilder
        },
    },
    rand::Rand,
    utilities::shuffle,
};

type ElementType = f32;

const IMAGE_ROW_SIZE: usize = 28;
const IMAGE_DOT_VALUE: usize = IMAGE_ROW_SIZE * IMAGE_ROW_SIZE;
const IMAGE_CHANNEL_VALUE: usize = 1;

pub fn test() -> Result<(), Box<dyn std::error::Error>> {
    let mini_batch_sample_size: usize = 125;

    let training_value: u32 = 60000;
    let validation_value = 9000;
    let test_value = 1000;

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

    let validation_chunk_size = 100usize;
    let test_chunk_size = 100usize;

    let mut r = Rand::new();

    // ニューラルネットワークの作成
    let mut tree = NNDataFlowTree::new();

    let conv1 = ConvolutionBuilder::new()
        .input_channel_value(1)
        .input_image_size((IMAGE_ROW_SIZE, IMAGE_ROW_SIZE))
        .filter_size((3, 3))
        .filter_value(2)
        .padding(1)
        .build();
    let batch1 = BatchNorm2dBuilder::new().channel_size(2).build();
    let relu_c1 = ReLUBuilder::new().build();
    let pool1 = MaxPoolingBuilder::new()
        .window_size((2, 2))
        .stride(2)
        .build();
    let conv2 = ConvolutionBuilder::new()
        .input_channel_value(2)
        .filter_value(4)
        .input_image_size((14, 14))
        .filter_size((3, 3))
        .padding(1)
        .build();
    let batch2 = BatchNorm2dBuilder::new().channel_size(4).build();
    let relu_c2 = ReLUBuilder::new().build();
    let pool2 = MaxPoolingBuilder::new()
        .window_size((2, 2))
        .stride(2)
        .build();
    let conv3 = ConvolutionBuilder::new()
        .input_channel_value(4)
        .input_image_size((7, 7))
        .filter_size((3, 3))
        .filter_value(8)
        .build();
    let batch3 = BatchNorm2dBuilder::new().channel_size(8).build();
    let relu_c3 = ReLUBuilder::new().build();

    let reshape = ReshapeTensorBuilder::new()
        .shape(vec![mini_batch_sample_size, 200])
        .build();

    let linear1 = LinearBuilder::new()
        .input_node_value(200)
        .output_node_value(40)
        .build();
    let relu1 = ReLUBuilder::new().build();
    let linear2 = LinearBuilder::new()
        .input_node_value(40)
        .output_node_value(10)
        .build();

    let softmax_and_celoss = SoftmaxAndCELossBuilder::new().build();

    let conv_info1 = &tree.add_from_root(conv1);
    let batch_info1 = &tree.add(&conv_info1, batch1);
    let relu_c_info1 = &tree.add(&batch_info1, relu_c1);
    let pool_info1 = &tree.add(&relu_c_info1, pool1);
    let conv_info2 = &tree.add(&pool_info1, conv2);
    let batch_info2 = &tree.add(&conv_info2, batch2);
    let relu_c_info2= &tree.add(&batch_info2, relu_c2);
    let pool_info2 = &tree.add(&relu_c_info2, pool2);
    let conv_info3 = &tree.add(&pool_info2, conv3);
    let batch_info3 = &tree.add(&conv_info3, batch3);
    let relu_c_info3= &tree.add(&batch_info3, relu_c3);

    let reshape_info = &tree.add(&relu_c_info3, reshape);

    let linear_info1 = &tree.add(&reshape_info, linear1);
    let relu_info1 = &tree.add(&linear_info1, relu1);
    let linear_info2 = &tree.add(&relu_info1, linear2);
    let _ = &tree.add(&linear_info2, softmax_and_celoss);

    // lr range test
    let mut training_rate: ElementType = 1e-7;
    let rate_up_ratio: ElementType = 1.01;
    let finish_rate: ElementType = 1e-1;
    let mut iter_count = 0;

    let mut is_default = true;
    let mut first_error: ElementType = 0.0;

    let mut result_string = "".to_string();

    let expected_iter_times = finish_rate.log(rate_up_ratio) - training_rate.log(rate_up_ratio);
    println!("expected iteration times: {}", expected_iter_times);

    'outer: while training_rate < finish_rate {
        let shuffle_index = shuffle::generate_shuffle_array(training_value as usize, &mut r);
        let mut error: ElementType = 0.0;

        let mut mini_batch_count = 0;

        for indices in shuffle_index.chunks_exact(mini_batch_sample_size as usize) {
            mini_batch_count += 1;
            let (inputs, expects) = make_mini_batch_dataset(indices, &mnist);

            error = tree
                .forward(
                    &NNForwardInput::<ElementType> {
                        inputs: vec![inputs.view().into_dyn()],
                        target: Some(expects.view().into_dyn()),
                    },
                    true,
                )
                .unwrap()
                .into_dimensionality::<Ix0>()
                .unwrap()
                .into_scalar();
            _ = &tree.propagate_grad(None, training_rate);

            let average_error = error / mini_batch_sample_size as ElementType;

            result_string += &format!("{}\t{}\n", training_rate, average_error);

            iter_count += 1;
            print!("\r{}", iter_count);
            std::io::stdout().flush().unwrap();

            if is_default {
                first_error = average_error;
                is_default = false;
            }

            if training_rate > finish_rate || average_error > first_error * 1.5 {
                println!(" !end this point");
                println!(
                    "last_rate: {}\nfirst_error: {}\nlast_error: {}",
                    training_rate, first_error, average_error
                );
                break 'outer;
            }

            training_rate *= rate_up_ratio;
        }
    }

    // ファイル出力
    let mut file = File::create("src/lr_range_test_results/test.txt")?;
    // let buf = BufReader::new(io::stdin()).lines().collect::<io::Result<Vec<String>>>()?.join("\n");
    write!(file, "{}", result_string)?;
    file.flush()?;
    Ok(())
}

fn make_mini_batch_dataset<'a>(
    indices: &'a [usize],
    dataset: &'a NormalizedMnist,
) -> (Array4<ElementType>, Array2<ElementType>) {
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

    let inputs = Array4::<ElementType>::from_shape_vec(
        (
            sample_size,
            IMAGE_CHANNEL_VALUE,
            IMAGE_ROW_SIZE,
            IMAGE_ROW_SIZE,
        ),
        inputs_array,
    )
    .unwrap();
    let expects = Array2::<ElementType>::from_shape_vec((sample_size, 10), expected_array).unwrap();

    (inputs.to_owned(), expects.to_owned())
}
