use std::any::Any;

use ndarray::{Array, Array1, Array2, Array4, ArrayD, Axis};

use crate::cnn_information::{
    ConvolutionInformation, FullConnectedInformation, LayerInformation, LayerInformationContent, LayerType, OutputInformation, OutputType, PoolingInformation
};

pub struct NeuralNetworkCNN {
    pub output_type: OutputType,
    pub layers_information: Vec<LayerInformation>,
    pub convolution_filters: Vec<Array4<f64>>,
    pub convolution_biases: Vec<Array1<f64>>,
    pub full_connected_weights: Vec<Array2<f64>>,
    pub full_connected_biases: Vec<Array1<f64>>,
    pub convolution_values: Vec<Array4<f64>>,
    pub convolution_after_activation: Vec<Array4<f64>>,
    pub fc_values: Vec<Array2<f64>>,
    pub fc_after_activation: Vec<Array2<f64>>,
}

impl NeuralNetworkCNN {
    pub fn forward(&mut self, inputs: Array4<f64>, expects: Array4<f64>) {}

    pub fn view_layers_information_state(&self) {
        dbg!(&self.layers_information);
    }

    fn create_full_connected_layer(batch_size: usize, input_node_value: usize, info: &FullConnectedInformation) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        let weight = Array2::zeros([input_node_value, info.node_value]);
        let bias = Array2::zeros([batch_size, info.node_value]);

        (weight, bias.clone(), bias)
    }

    fn create_output_layer(batch_size: usize, input_node_value: usize, info: &OutputInformation) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        let weight = Array2::zeros([input_node_value, info.node_value]);
        let bias = Array2::zeros([batch_size, info.node_value]);

        (weight, bias.clone(), bias)
    }
}
