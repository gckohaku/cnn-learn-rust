use std::result;

use ndarray::{Array2, Array4};

use crate::{
    cnn_information::{LayerInformation, OutputInformation},
    convolution_network::ConvolutionNetwork,
    fully_connected_network::FullyConnectedNetwork,
};

pub struct NeuralNetworkCNN {
    fcnn: FullyConnectedNetwork,
    cnn: ConvolutionNetwork,
    convolution_last_shape: [usize; 4],
    // layers_information: Vec<LayerInformation>,
    // output_information: OutputInformation,
}

impl NeuralNetworkCNN {
    pub fn new(
        batch_size: usize,
        input_channel_value: usize,
        input_image_size: (usize, usize),
        fc_nodes_values: Vec<usize>,
        cnn_layers_information: Vec<LayerInformation>,
        output_information: OutputInformation,
    ) -> Self {
        let mut cnn = ConvolutionNetwork::new(
            batch_size,
            input_channel_value,
            input_image_size,
            cnn_layers_information.clone(),
        );
        let mut fcnn =
            FullyConnectedNetwork::new(batch_size, fc_nodes_values, output_information.clone());

        Self {
            fcnn,
            cnn,
            convolution_last_shape: [0, 0, 0, 0],
            // layers_information: cnn_layers_information,
            // output_information,
        }
    }

    pub fn forward(&mut self, inputs: &Array4<f64>, expects: &Array2<f64>) {
        let inputs_shape = inputs.shape();
        let batch_size = inputs_shape[0];

        let convolution_result = self.cnn.forward(&inputs);
        let result_shape = convolution_result.shape();
        self.convolution_last_shape = [batch_size, result_shape[1], result_shape[2], result_shape[3]];
        let spread_result = convolution_result
            .to_shape((
                batch_size,
                result_shape[1] * result_shape[2] * result_shape[3],
            ))
            .unwrap()
            .to_owned();

        self.fcnn.forward(&spread_result, &expects);
    }

    pub fn backward(&mut self, expects: &Array2<f64>, eta: f64) {
        self.fcnn.backward(expects, eta);
        let gradient = self.fcnn.get_input_gradient();

        let shaped_gradient = gradient.to_shape(self.convolution_last_shape).unwrap().to_owned();
        self.cnn.backward(&shaped_gradient, eta);
    }

    pub fn get_error(&self) -> f64 {
        self.fcnn.get_error()
    }

    // pub fn view_layers_information_state(&self) {
    //     dbg!(&self.layers_information);
    // }

    // fn create_full_connected_layer(batch_size: usize, input_node_value: usize, info: &FullConnectedInformation) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    //     let weight = Array2::zeros([input_node_value, info.node_value]);
    //     let bias = Array2::zeros([batch_size, info.node_value]);

    //     (weight, bias.clone(), bias)
    // }

    // fn create_output_layer(batch_size: usize, input_node_value: usize, info: &OutputInformation) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
    //     let weight = Array2::zeros([input_node_value, info.node_value]);
    //     let bias = Array2::zeros([batch_size, info.node_value]);

    //     (weight, bias.clone(), bias)
    // }
}
