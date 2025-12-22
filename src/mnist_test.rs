use mnist::{MnistBuilder, NormalizedMnist};
use ndarray::{Array2, Array4};

use crate::{rand::Rand, utilities::shuffle};

const IMAGE_ROW_SIZE: usize = 28;
const IMAGE_DOT_VALUE: usize = IMAGE_ROW_SIZE * IMAGE_ROW_SIZE;
const IMAGE_CHANNEL_VALUE: usize = 1;

pub fn mnist_process() {
    let epoch_value = 10;
	let mini_batch_sample_size = 250;

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

    let mut r = &mut Rand::new();

    for epoch in 1..=epoch_value {
        let shuffle_index = shuffle::generate_shuffle_array(training_value as usize, &mut r);
        let mut epoch_error = 0.0;

        let mut mini_batch_count = 0;

        for indexes in shuffle_index.chunks(mini_batch_sample_size) {

        }
    }
}

fn make_mini_batch_dataset(indices: &Vec<usize>, dataset: &NormalizedMnist) {
    let sample_size = indices.len();

    let inputs = Array4::<f64>::zeros((0, IMAGE_CHANNEL_VALUE, IMAGE_ROW_SIZE, IMAGE_ROW_SIZE));
    let expects = Array2::<f64>::zeros((0, 10));

    for index in indices {

    }
}