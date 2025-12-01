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

    fn create_convolution_layer(
        batch_size: usize,
        input_channel_value: usize,
        input_image_size: (usize, usize),
        info: &ConvolutionInformation,
    ) -> (Array4<f64>, Array1<f64>, Array4<f64>) {
        let output_channel_value = info.filter_value;
        let stride = info.stride;
        let padding = info.padding;
        let filter_size = info.filter_size;
        let (h, w) = input_image_size;
        let output_image_size = (
            ((h - filter_size.0 + 2 * padding) as f64 / stride as f64).floor() as usize + 1,
            ((w - filter_size.1 + 2 * padding) as f64 / stride as f64).floor() as usize + 1,
        );

        // フィルター生成
        let weight = Array4::zeros([
            info.filter_value,
            input_channel_value,
            filter_size.0,
            filter_size.1,
        ]);
        // バイアス生成
        let bias = Array1::zeros([info.filter_value]);
        // 出力テンソル生成
        let output = Array4::zeros([
            batch_size,
            output_channel_value,
            output_image_size.0,
            output_image_size.1,
        ]);

        (weight, bias, output)
    }

    fn create_pooling_layer(
        batch_size: usize,
        input_channel_value: usize,
        input_image_size: (usize, usize),
        info: &PoolingInformation,
    ) -> (Array4<f64>, Array4<f64>) {
        let output_channel_value = input_channel_value;
        let window_size = info.window_size;
        let output_image_size = (
            input_image_size.0 / window_size.0,
            input_image_size.1 / window_size.1,
        );

        let output = Array4::zeros([batch_size, output_channel_value, output_image_size.0, output_image_size.1]);

        (output.clone(), output)
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
