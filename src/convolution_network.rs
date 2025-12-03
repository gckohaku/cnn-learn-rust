use ndarray::{Array1, Array2, Array4, Dimension};

use crate::cnn_information::{
    ConvolutionInformation, LayerInformation, LayerType, OutputType, PoolingInformation,
};

#[derive(Debug, Clone)]
pub struct ConvolutionNetwork {
    pub layers_information: Vec<LayerInformation>,
    pub filters: Vec<Array4<f64>>,
    pub biases: Vec<Array1<f64>>,
    pub windows: Vec<Array2<f64>>,
    pub switches: Vec<Array2<(usize, usize)>>,
    pub values: Vec<Array4<f64>>,
    pub im2col_values: Vec<Array2<f64>>,
    pub values_after_activation: Vec<Array4<f64>>,
}

impl ConvolutionNetwork {
    pub fn new(
        batch_size: usize,
        input_channel_value: usize,
        input_image_size: (usize, usize),
        layers_information: Vec<LayerInformation>,
    ) -> Self {
        let mut filters = Vec::<Array4<f64>>::new();
        let mut biases = Vec::<Array1<f64>>::new();
        let mut windows = Vec::<Array2<f64>>::new();
        let mut switches = Vec::<Array2<(usize, usize)>>::new();
        let mut values = Vec::<Array4<f64>>::new();
        let im2col_values = Vec::<Array2<f64>>::new();
        let mut values_after_activation = Vec::<Array4<f64>>::new();

        let mut before_image_shape = [
            batch_size,
            input_channel_value,
            input_image_size.0,
            input_image_size.1,
        ];

        for i in 0..layers_information.len() {
            let input_size = (before_image_shape[2], before_image_shape[3]);

            if layers_information[i].layer_type == LayerType::Convolution {
                let info: &ConvolutionInformation =
                    layers_information[i].information.as_convolution().unwrap();

                let (weight, bias, value) = Self::create_convolution_layer(
                    batch_size,
                    before_image_shape[1],
                    input_size,
                    info,
                );

                filters.push(weight);
                biases.push(bias);
                values.push(value.clone());
                values_after_activation.push(value);
            } else if layers_information[i].layer_type == LayerType::Pooling {
                let info: &PoolingInformation =
                    layers_information[i].information.as_pooling().unwrap();

                let (window, switch, output) =
                    Self::create_pooling_layer(batch_size, before_image_shape[1], input_size, info);

                windows.push(window);
                switches.push(switch);
                values.push(output.clone());
                values_after_activation.push(output);
            }

            before_image_shape = values.last().unwrap().shape().try_into().unwrap();
        }

        Self {
            layers_information,
            filters,
            biases,
            windows,
            switches,
            values,
            im2col_values,
            values_after_activation,
        }
    }

    pub fn forward(&mut self, inputs: Array4<f64>) -> Array4<f64> {
        self.values[0] = inputs.clone();
        self.values_after_activation[0] = inputs.clone();
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
    ) -> (Array2<f64>, Array2<(usize, usize)>, Array4<f64>) {
        let output_channel_value = input_channel_value;
        let window_size = info.window_size;
        let output_image_size = (
            input_image_size.0 / window_size.0,
            input_image_size.1 / window_size.1,
        );

        let window = Array2::zeros([window_size.0, window_size.1]);

        let switch = Array2::from_elem([window_size.0, window_size.1], (0, 0));

        let output = Array4::zeros([
            batch_size,
            output_channel_value,
            output_image_size.0,
            output_image_size.1,
        ]);

        (window, switch, output)
    }
}
