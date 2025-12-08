use ndarray::{Array3, Array4};

use crate::{
    cnn_information::{
        ActivationType, ConvolutionInformation, LayerInformation, LayerInformationContent,
        LayerType,
    },
    convolution_network::ConvolutionNetwork,
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
    let mut cnn = ConvolutionNetwork::new(
        3,
        3,
        (4, 4),
        vec![LayerInformation {
            layer_type: LayerType::Convolution,
            information: LayerInformationContent::Convolution(ConvolutionInformation {
                filter_size: (2, 2),
                filter_value: 3,
                stride: 1,
                padding: 0,
                activation_type: ActivationType::ReLU,
            }),
        }],
    );

	let range_vec = (1..(3*3*4*4+1)).map(|x| x as f64 / 127.5 - 1.0).collect();
	let mut inputs = Array4::from_shape_vec((3, 3, 4, 4), range_vec).unwrap();

	cnn.forward(&inputs);
}
