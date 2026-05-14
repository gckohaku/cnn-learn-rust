use chrono::Local;
use mnist::{MnistBuilder, NormalizedMnist};
use ndarray::{Array2, Array4, Ix0};
use std::{f32::consts::PI, fs::File, io::Write, time};

use crate::{
    nn_modules::{
        NNDataFlowTree, NNForwardInput, NNModule, NNModuleBuilder, ReshapeTensor, SoftmaxAndCELoss,
        builders::{
            BatchNorm2dBuilder, ConvolutionBuilder, LinearBuilder, MaxPoolingBuilder, ReLUBuilder,
            ReshapeTensorBuilder, SoftmaxAndCELossBuilder,
        },
    },
    rand::Rand,
    utilities::shuffle,
};

type ElementType = f32;

const IMAGE_ROW_SIZE: usize = 28;
const IMAGE_DOT_VALUE: usize = IMAGE_ROW_SIZE * IMAGE_ROW_SIZE;
const IMAGE_CHANNEL_VALUE: usize = 1;

pub fn mnist_process() -> Result<(), Box<dyn std::error::Error>> {
    let epoch_value = 61;
    let mini_batch_sample_size: usize = 125;

    let training_value: u32 = 1000;
    let validation_value = 1000;
    let test_value = 1000;

    let mini_batches_per_epoch = training_value / mini_batch_sample_size as u32;

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
        .is_bias(false)
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
        .is_bias(false)
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
        .is_bias(false)
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
    let relu_c_info2 = &tree.add(&batch_info2, relu_c2);
    let pool_info2 = &tree.add(&relu_c_info2, pool2);
    let conv_info3 = &tree.add(&pool_info2, conv3);
    let batch_info3 = &tree.add(&conv_info3, batch3);
    let relu_c_info3 = &tree.add(&batch_info3, relu_c3);

    let reshape_info = &tree.add(&relu_c_info3, reshape);

    let linear_info1 = &tree.add(&reshape_info, linear1);
    let relu_info1 = &tree.add(&linear_info1, relu1);
    let linear_info2 = &tree.add(&relu_info1, linear2);
    let output_info = &tree.add(&linear_info2, softmax_and_celoss);

    let mut writing_file_string: String =
        "epoch\tlearning rate\tlearning error\tvalidation error\tvalidation collect rate\n"
            .to_string();

    // 処理時間計測用
    let epochs_now = time::Instant::now();

    let max_learning_rate = 2.5e-4;
    let min_learning_rate = 5e-6;

    for epoch in 1..=epoch_value {
        let shuffle_index = shuffle::generate_shuffle_array(training_value as usize, &mut r);
        let mut epoch_error = 0.0;

        let mut mini_batch_count = 0;
        // let epoch_learning_rate = max_learning_rate + ((((epoch - 1) % 10) as ElementType / 9.0) * (min_learning_rate - max_learning_rate));
        let epoch_learning_rate = if (epoch - 1) % 20 <= 5 {
            min_learning_rate + ((max_learning_rate - min_learning_rate) * (epoch - 1) as ElementType / 5.0)
        } else {
            min_learning_rate
                + ((max_learning_rate - min_learning_rate) * ((((((epoch - 6) % 20) % 15) as ElementType / 15.0) * PI).cos() + 1.0)
                    / 2.0)
        };

        println!("learning rate: {}", epoch_learning_rate);

        for indices in shuffle_index.chunks_exact(mini_batch_sample_size as usize) {
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
                .unwrap()
                .into_dimensionality::<Ix0>()
                .unwrap()
                .into_scalar();
            _ = &tree.propagate_grad(None, epoch_learning_rate);

            print!(
                "\rmini batch count: {} / {}",
                mini_batch_count, mini_batches_per_epoch
            );
            std::io::stdout().flush().unwrap();
        }

        println!(
            "\nepoch {:6} error: {:13.10}",
            epoch,
            epoch_error / training_value as ElementType
        );

        writing_file_string += &format!(
            "{}\t{}\t{}\t",
            epoch,
            epoch_learning_rate,
            epoch_error / training_value as ElementType
        );

        // 検証部分
        println!("validation test:");

        epoch_error = 0.0;

        let reshape_tensor: &mut ReshapeTensor<ElementType> = tree
            .access_module_mut(&reshape_info)
            .as_any_mut()
            .downcast_mut()
            .unwrap();
        reshape_tensor.change_shape(vec![validation_chunk_size, 200]);

        let out: &mut SoftmaxAndCELoss<ElementType> = &mut tree
            .access_module_mut(&output_info)
            .as_any_mut()
            .downcast_mut()
            .unwrap();
        out.is_test = true;
        out.test_correct_value = 0usize;

        for indices in (0..validation_value)
            .map(|x| x as usize)
            .collect::<Vec<usize>>()
            .chunks(validation_chunk_size as usize)
        {
            let (inputs, expects) = make_validation_data_set(indices, &mnist);

            epoch_error += &tree
                .forward(
                    &NNForwardInput::<ElementType> {
                        inputs: vec![inputs.view().into_dyn()],
                        target: Some(expects.view().into_dyn()),
                    },
                    false,
                )
                .unwrap()
                .into_dimensionality::<Ix0>()
                .unwrap()
                .into_scalar();
        }

        let reshape_tensor: &mut ReshapeTensor<ElementType> = tree
            .access_module_mut(&reshape_info)
            .as_any_mut()
            .downcast_mut()
            .unwrap();
        reshape_tensor.change_shape(vec![mini_batch_sample_size, 200]);

        let out: &mut SoftmaxAndCELoss<ElementType> = &mut tree
            .access_module_mut(&output_info)
            .as_any_mut()
            .downcast_mut()
            .unwrap();
        let result = out.test_correct_value;
        out.is_test = false;

        println!(
            "collect rate: {}",
            result as ElementType / validation_value as ElementType
        );
        println!("              ({} / {})", result, validation_value);
        println!("error: {}\n", epoch_error / validation_value as ElementType);

        writing_file_string += &format!(
            "{}\t{}\n",
            epoch_error / validation_value as ElementType,
            result as ElementType / validation_value as ElementType
        );
    }

    println!("after leaning test:");

    let reshape_tensor: &mut ReshapeTensor<ElementType> = tree
        .access_module_mut(&reshape_info)
        .as_any_mut()
        .downcast_mut()
        .unwrap();
    reshape_tensor.change_shape(vec![test_chunk_size, 200]);

    let out: &mut SoftmaxAndCELoss<ElementType> = &mut tree
        .access_module_mut(&output_info)
        .as_any_mut()
        .downcast_mut()
        .unwrap();
    out.is_test = true;
    out.test_correct_value = 0usize;

    let mut test_error = 0.0;

    for indices in (0..test_value)
        .map(|x| x as usize)
        .collect::<Vec<usize>>()
        .chunks(validation_chunk_size as usize)
    {
        let (inputs, expects) = make_test_data_set(indices, &mnist);

        test_error += &tree
            .forward(
                &NNForwardInput::<ElementType> {
                    inputs: vec![inputs.view().into_dyn()],
                    target: Some(expects.view().into_dyn()),
                },
                false,
            )
            .unwrap()
            .into_dimensionality::<Ix0>()
            .unwrap()
            .into_scalar();
    }

    let out: &mut SoftmaxAndCELoss<ElementType> = &mut tree
        .access_module_mut(&output_info)
        .as_any_mut()
        .downcast_mut()
        .unwrap();
    let result = out.test_correct_value;
    out.is_test = false;

    println!(
        "collect rate: {}",
        result as ElementType / test_value as ElementType
    );
    println!("              ({} / {})", result, test_value);
    println!("error: {}\n", test_error / test_value as ElementType);

    writing_file_string += "\nafter learning test\n";
    writing_file_string += &format!(
        "\tcollect rate\t{}\n",
        result as ElementType / test_value as ElementType
    );
    writing_file_string += &format!("\terror\t{}\n", test_error / test_value as ElementType);

    let epochs_process_duration = epochs_now.elapsed().as_secs_f64();

    // 処理時間表示
    println!("epochs process duration: {}sec.", epochs_process_duration);

    writing_file_string += &format!("\nepochs process duration\t{}sec.", epochs_process_duration);

    // ファイル出力
    let mut file =
        File::create("src/learning_results/".to_string() + &Local::now().to_string() + &".txt")?;
    // let buf = BufReader::new(io::stdin()).lines().collect::<io::Result<Vec<String>>>()?.join("\n");
    write!(file, "{}", writing_file_string)?;
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

fn make_validation_data_set<'a>(
    indices: &'a [usize],
    dataset: &'a NormalizedMnist,
) -> (Array4<ElementType>, Array2<ElementType>) {
    let sample_size = indices.len();

    let mut inputs_array = Vec::<ElementType>::new();
    let mut expected_array = Vec::<ElementType>::new();

    for index in indices {
        let val_data: &Vec<ElementType> = &dataset.val_img
            [((*index) * IMAGE_DOT_VALUE)..(((*index) + 1) * IMAGE_DOT_VALUE)]
            .iter()
            .map(|&x| x as ElementType)
            .collect();
        inputs_array.extend_from_slice(val_data);

        let val_label: &Vec<ElementType> = &dataset.val_lbl[((*index) * 10)..(((*index) + 1) * 10)]
            .iter()
            .map(|&x| x as ElementType)
            .collect();
        expected_array.extend_from_slice(val_label);
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

fn make_test_data_set<'a>(
    indices: &'a [usize],
    dataset: &'a NormalizedMnist,
) -> (Array4<ElementType>, Array2<ElementType>) {
    let sample_size = indices.len();

    let mut inputs_array = Vec::<ElementType>::new();
    let mut expected_array = Vec::<ElementType>::new();

    for index in indices {
        let tst_data: &Vec<ElementType> = &dataset.tst_img
            [((*index) * IMAGE_DOT_VALUE)..(((*index) + 1) * IMAGE_DOT_VALUE)]
            .iter()
            .map(|&x| x as ElementType)
            .collect();
        inputs_array.extend_from_slice(tst_data);

        let tst_label: &Vec<ElementType> = &dataset.tst_lbl[((*index) * 10)..(((*index) + 1) * 10)]
            .iter()
            .map(|&x| x as ElementType)
            .collect();
        expected_array.extend_from_slice(tst_label);
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
