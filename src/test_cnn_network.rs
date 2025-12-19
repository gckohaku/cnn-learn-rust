use ndarray::{Array2, Array3, Array4, array, s};

use crate::{
    cnn_information::{
        ActivationType, ConvolutionInformation, LayerInformation, LayerInformationContent,
        LayerType, OutputInformation, OutputType, PoolingInformation, PoolingType,
    },
    cnn_network::NeuralNetworkCNN,
    rand::Rand,
    simple_dataset::dataset::{SimpleDataset, SmallData},
    utilities::shuffle,
};

pub fn check_cnn() {
    let mut nn = NeuralNetworkCNN::new(
        5,
        3,
        (5, 5),
        vec![24, 8, 3],
        vec![
            LayerInformation {
                layer_type: LayerType::Convolution,
                information: LayerInformationContent::Convolution(ConvolutionInformation {
                    filter_size: (2, 2),
                    filter_value: 6,
                    stride: 1,
                    padding: 0,
                    activation_type: ActivationType::ReLU,
                }),
            },
            LayerInformation {
                layer_type: LayerType::Pooling,
                information: LayerInformationContent::Pooling(PoolingInformation {
                    window_size: (2, 2),
                    stride: 1,
                    pooling_type: PoolingType::MaxPooling,
                }),
            },
        ],
        OutputInformation {
            node_value: 3,
            output_type: OutputType::MultiClassClassification,
        },
    );

    let mut r = Rand::new();
    let dataset = SimpleDataset::new();

    let epoch_value = 10;
    let mini_batch_sample_value = 5;

    for epoch in 1..=epoch_value {
        let shuffled_indices = shuffle::generate_shuffle_array(30, &mut r);
        let mut epoch_error = 0.0;

        for indices in shuffled_indices.chunks(mini_batch_sample_value) {
            let batch_size = indices.len();

            let mut batch_array = Vec::new();
            let mut expect_array = Vec::new();

            for index in indices {
                let current_batch = &dataset.data[*index];

                batch_array.push(current_batch.image);
                expect_array.push(current_batch.label);
            }

            let mut batch_data = Array4::from(batch_array);
            let mut expect_data = Array2::from(expect_array);

            // for i in 0..batch_size {
            //     batch_data
            //         .slice_mut(s![i, .., .., ..])
            //         .assign(&array![batch[i].image].slice(s![0, .., .., ..]));
            //     expect_data
            //         .slice_mut(s![i, ..])
            //         .assign(&array![batch[i].label].slice(s![0, ..]));
            // }

            nn.forward(&batch_data, &expect_data);
            epoch_error += nn.get_error();
            nn.backward(&expect_data, 0.001);

            println!("epoch {:3} error: {:13.10}", epoch, epoch_error / 30.0)
        }
    }
}
