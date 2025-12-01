use ndarray::{Array1, Array4, Dimension};

use crate::cnn_information::{ConvolutionInformation, LayerInformation, LayerType, OutputType, PoolingInformation};

pub struct ConvolutionNetwork {
    pub layers_information: Vec<LayerInformation>,
    pub filters: Vec<Array4<f64>>,
    pub biases: Vec<Array1<f64>>,
    pub values: Vec<Array4<f64>>,
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
        let mut values = Vec::<Array4<f64>>::new();
        let mut values_after_activation = Vec::<Array4<f64>>::new();

		let mut before_image_shape = [batch_size, input_channel_value, input_image_size.0, input_image_size.1];

        for i in 0..layers_information.len() {
            if layers_information[i].layer_type == LayerType::Convolution {
                let image_shape = values[i - 1].shape();
                let info: &ConvolutionInformation =
                    layers_information[i].information.as_convolution().unwrap();
                let output_image_height =
                    (before_image_shape[2] - info.filter_size.0 + 2 * info.padding) / info.stride + 1;
                let output_image_width =
                    (before_image_shape[3] - info.filter_size.1 + 2 * info.padding) / info.stride + 1;

                let (weight, bias, value) = Self::create_convolution_layer(
                    batch_size,
                    input_channel_value,
                    input_image_size,
                    info,
                );

				before_image_shape = value.shape();

				filters.push(weight);
				biases.push(bias);
				values.push(value.clone());
				values_after_activation.push(value);


            }
			else if layers_information[i].layer_type == LayerType::Pooling {

			}
        }

        Self {
            layers_information,
            filters,
            biases,
            values,
            values_after_activation,
        }
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
}
