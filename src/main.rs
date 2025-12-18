use ndarray::{Array2, Array3, Array4, arr2, array};

use crate::{
    cnn_information::{
        ActivationType, ConvolutionInformation, LayerInformation, LayerInformationContent,
        LayerType, OutputInformation, OutputType, PoolingInformation, PoolingType,
    },
    cnn_transformations::im2col::im2col_for_pooling,
    convolution_network::ConvolutionNetwork,
    fully_connected_network::FullyConnectedNetwork, rand::Rand,
};

mod cnn_activations;
mod cnn_information;
mod cnn_network;
mod cnn_transformations;
mod convolution_network;
mod fully_connected_network;
mod rand;
mod type_utilities;
mod simple_dataset;
mod test_cnn_network;

fn main() {
    test_cnn_network::test();
}
