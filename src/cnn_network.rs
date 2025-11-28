use std::any::Any;

use ndarray::{Array1, Array4, ArrayD};

use crate::cnn_information::{
    ConvolutionInInformation, LayerInformation, LayerType, PoolingInformation,
};

pub struct NeuralNetworkCNN {
    pub layers_information: Vec<LayerInformation>,
    pub weights: Vec<ArrayD<f64>>,
    pub biases: Vec<Array1<f64>>,
    pub values: Vec<ArrayD<f64>>,
    pub values_after_activation: Vec<ArrayD<f64>>,
}

impl NeuralNetworkCNN {
    pub fn new(
        batch_size: usize,
        input_channel_value: usize,
        input_image_size: (usize, usize),
        layers_information: Vec<LayerInformation>,
    ) -> Self {
        let mut weights = Vec::<ArrayD<f64>>::new();
        let mut biases = Vec::<Array1<f64>>::new();
        let mut values = Vec::<ArrayD<f64>>::new();
        let mut values_after_activation = Vec::<ArrayD<f64>>::new();

        for i in 0..layers_information.len() {
            if layers_information[i].layer_type == LayerType::Convolution {
                let image_shape = values[i - 1].shape();
                // let input_image_height =
                let outputImageHeight = () + 1;
            }
        }

        Self {
            layers_information,
            weights,
            biases,
            values,
            values_after_activation,
        }
    }

    pub fn forward(&mut self, inputs: Array4<f64>, expects: Array4<f64>) {}

    pub fn view_layers_information_state(&self) {
        dbg!(&self.layers_information);
    }

    fn create_convolution_layer(
        batch_size: usize,
        input_channel_value: usize,
        input_image_size: (usize, usize),
        info: ConvolutionInInformation,
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
        info: PoolingInformation,
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
}
