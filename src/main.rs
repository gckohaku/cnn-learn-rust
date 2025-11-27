use ndarray::{Array, Array2, Array4};

use crate::{
    cnn_information::{
        ActivationType, ConvolutionInInformation, FullConnectedInformation, LayerInformation, LayerInformationContent, LayerType, OutputInformation, OutputType, PoolingInformation, PoolingType
    },
    cnn_network::NeuralNetworkCNN,
    cnn_transformations::im2col::im2col,
};

mod cnn_information;
mod cnn_network;
mod cnn_transformations;
mod cnn_activations;
mod type_utilities;

fn main() {
    let cnn = NeuralNetworkCNN::new(
        1,
        1,
        (28, 28),
        vec![LayerInformation {
            layer_type: LayerType::Convolution,
            information: LayerInformationContent::Convolution(ConvolutionInInformation {
                filter_size: (3, 3),
                filter_value: 8,
                stride: 1,
                padding: 0,
                activation_type: ActivationType::ReLU,
            }),
        },
        LayerInformation {
            layer_type: LayerType::Pooling,
            information: LayerInformationContent::Pooling(PoolingInformation {
                window_size: (2, 2),
                pooling_type: PoolingType::MaxPooling,
            })
        },
        LayerInformation {
            layer_type: LayerType::FullConnected,
            information: LayerInformationContent::FullConnected(FullConnectedInformation {
                node_value: 100,
                activation_type: ActivationType::ReLU,
            })
        },
        LayerInformation {
            layer_type: LayerType::Output,
            information: LayerInformationContent::Output(OutputInformation {
                node_value: 10,
                output_type: OutputType::MultiClassClassification,
            })
        }],
    );

    cnn.view_layers_information_state();
}
