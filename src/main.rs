use ndarray::{Array2, Array3, Array4, arr2, array};

use crate::{
    cnn_information::{
        ActivationType, ConvolutionInformation, LayerInformation, LayerInformationContent,
        LayerType, OutputInformation, OutputType,
    },
    convolution_network::ConvolutionNetwork,
    fully_connected_network::FullyConnectedNetwork,
    rand::Rand,
};

mod cnn_activations;
mod cnn_information;
mod cnn_network;
mod cnn_transformations;
mod convolution_network;
mod fully_connected_network;
mod rand;
mod type_utilities;

fn main() {
    // let mut cnn = ConvolutionNetwork::new(
    //     3,
    //     3,
    //     (4, 4),
    //     vec![LayerInformation {
    //         layer_type: LayerType::Convolution,
    //         information: LayerInformationContent::Convolution(ConvolutionInformation {
    //             filter_size: (2, 2),
    //             filter_value: 3,
    //             stride: 1,
    //             padding: 0,
    //             activation_type: ActivationType::ReLU,
    //         }),
    //     }],
    // );

    // let range_vec = (1..(3*3*4*4+1)).map(|x| x as f64 / 127.5 - 1.0).collect();
    // let mut inputs = Array4::from_shape_vec((3, 3, 4, 4), range_vec).unwrap();

    // cnn.forward(&inputs);

    let mut fcnn = FullyConnectedNetwork::new(
        3,
        vec![6, 4],
        OutputInformation {
            node_value: 4,
            output_type: OutputType::MultiClassClassification,
        },
    );

    let inputs = Array2::<f64>::from(vec![
        [
            0.5663851872,
            0.1837966773,
            0.2425344167,
            0.3741499325,
            -0.8060948714,
            -0.4273705264,
        ],
        [
            -0.3697271179,
            -0.9031432606,
            -0.9239777229,
            0.6942769596,
            -0.3384569502,
            -0.2937747911,
        ],
        [
            -0.0557447956,
            0.5460896469,
            0.9600364943,
            0.2319100103,
            0.3744701119,
            0.6146683330,
        ],
    ]);
    let expects = Array2::<f64>::from(vec![
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]);

    fcnn.forward(&inputs, &expects);
}
