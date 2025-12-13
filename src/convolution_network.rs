use core::f64;

use ndarray::{
    Array, Array1, Array2, Array4, Axis, Zip, parallel::prelude::IntoParallelRefIterator,
};

use crate::{
    cnn_information::{
        ConvolutionInformation, LayerInformation, LayerType, PoolingInformation, PoolingType,
    },
    cnn_transformations::im2col::{im2col, im2col_for_pooling},
    rand::Rand,
};

use ndarray::parallel::prelude::*;

#[derive(Debug, Clone)]
pub struct ConvolutionNetwork {
    pub layers_information: Vec<LayerInformation>,
    pub filters: Vec<Array4<f64>>,
    pub biases: Vec<Array1<f64>>,
    pub windows: Vec<Array2<f64>>,
    pub pooling_mask: Vec<Array1<(usize, usize)>>,
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
        let windows = Vec::<Array2<f64>>::new();
        let pooling_mask = Vec::<Array1<(usize, usize)>>::new();
        let values = Vec::<Array4<f64>>::new();
        let im2col_values = Vec::<Array2<f64>>::new();
        let values_after_activation = Vec::<Array4<f64>>::new();

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

                let (weight, bias, next_shape) = Self::create_convolution_layer(
                    batch_size,
                    before_image_shape[1],
                    input_size,
                    info,
                );

                filters.push(weight);
                biases.push(bias);

                before_image_shape = next_shape;
            } else if layers_information[i].layer_type == LayerType::Pooling {
                let info: &PoolingInformation =
                    layers_information[i].information.as_pooling().unwrap();

                let next_shape =
                    Self::create_pooling_layer(batch_size, before_image_shape[1], input_size, info);

                before_image_shape = next_shape;
            }
        }

        Self {
            layers_information,
            filters,
            biases,
            windows,
            pooling_mask,
            values,
            im2col_values,
            values_after_activation,
        }
    }

    pub fn forward(&mut self, inputs: &Array4<f64>) -> Array4<f64> {
        let mut convolution_count = 0;
        let mut pooling_count = 0;

        self.values.push(inputs.clone());
        self.values_after_activation.push(inputs.clone());

        for i in 0..self.layers_information.len() {
            let info = &self.layers_information[i];

            if info.layer_type == LayerType::Convolution {
                let convolution_info = info.information.as_convolution().unwrap();

                let input_shape = self.values_after_activation[i].shape();

                let (spread_image, spread_filter) = im2col(
                    &self.values_after_activation[i],
                    &self.filters[convolution_count],
                    convolution_info.stride,
                    convolution_info.padding,
                );

                let output_size = (
                    ((input_shape[2] - convolution_info.filter_size.0
                        + 2 * convolution_info.padding)
                        / convolution_info.stride)
                        + 1,
                    ((input_shape[3] - convolution_info.filter_size.1
                        + 2 * convolution_info.padding)
                        / convolution_info.stride)
                        + 1,
                );

                let bias_length = self.biases[convolution_count].len();
                let spread_result = spread_filter.dot(&spread_image)
                    + self.biases[convolution_count]
                        .to_shape((bias_length, 1))
                        .unwrap();

                let activated_spread_result: Array2<f64> = Array::from_shape_vec(
                    (
                        convolution_info.filter_value,
                        input_shape[0] * output_size.0 * output_size.1,
                    ),
                    spread_result.par_iter().map(|x| x.max(0.0)).collect(),
                )
                .unwrap();

                let reshape_result = &mut spread_result
                    .to_shape([
                        convolution_info.filter_value,
                        input_shape[0],
                        output_size.0,
                        output_size.1,
                    ])
                    .unwrap();

                let activated_reshape_result = &mut activated_spread_result
                    .to_shape([
                        convolution_info.filter_value,
                        input_shape[0],
                        output_size.0,
                        output_size.1,
                    ])
                    .unwrap();

                reshape_result.swap_axes(0, 1);
                activated_reshape_result.swap_axes(0, 1);

                self.im2col_values.push(activated_spread_result.to_owned());
                self.values.push(reshape_result.to_owned());
                self.values_after_activation
                    .push(activated_reshape_result.to_owned());

                convolution_count += 1;
            } else if info.layer_type == LayerType::Pooling {
                let pooling_info = info.information.as_pooling().unwrap();
                let input_shape = self.values_after_activation[i].shape();

                if pooling_info.pooling_type == PoolingType::MaxPooling {
                    let col_matrix = im2col_for_pooling(
                        &self.values_after_activation[i],
                        pooling_info.stride,
                        pooling_info.window_size,
                    );

                    let window_value = pooling_info.window_size.0 * pooling_info.window_size.1;

                    let spread_cols =
                        input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]
                            / window_value;

                    let mut mask = Array1::zeros(spread_cols);
                    let mut after_pooling = Array1::zeros(spread_cols);

                    Zip::from(after_pooling.view_mut())
                        .and(mask.view_mut())
                        .and(col_matrix.axis_iter(Axis(1)))
                        .par_for_each(|res, mask, col| {
                            let mut max_value = f64::NEG_INFINITY;
                            let mut max_index = 0;

                            for (i, &v) in col.iter().enumerate() {
                                if v > max_value {
                                    max_value = v;
                                    max_index = i;
                                }
                            }

                            *mask = max_index;
                            *res = max_value;
                        });

                        let reshape_pooling = after_pooling.to_shape((input_shape[0], input_shape[1], input_shape[2] / pooling_info.window_size.0, input_shape[3] / pooling_info.window_size.1)).unwrap();

                        self.values.push(reshape_pooling.to_owned());
                        self.values_after_activation.push(reshape_pooling.to_owned());
                }
            }
        }

        self.values_after_activation.last().unwrap().clone()
    }

    fn create_convolution_layer(
        batch_size: usize,
        input_channel_value: usize,
        input_image_size: (usize, usize),
        info: &ConvolutionInformation,
    ) -> (Array4<f64>, Array1<f64>, [usize; 4]) {
        let mut r = Rand::new();

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
        let mut weight = Array4::zeros([
            info.filter_value,
            input_channel_value,
            filter_size.0,
            filter_size.1,
        ]);
        // フィルターを He 初期化
        weight.mapv_inplace(|_x| {
            r.normal(
                0.0,
                (2.0 / (input_channel_value * filter_size.0 * filter_size.1) as f64).sqrt(),
            )
        });

        // バイアス生成
        let bias = Array1::zeros([info.filter_value]);
        // 出力テンソル生成
        // let output = Array4::zeros([
        //     batch_size,
        //     output_channel_value,
        //     output_image_size.0,
        //     output_image_size.1,
        // ]);

        (
            weight,
            bias,
            [
                batch_size,
                output_channel_value,
                output_image_size.0,
                output_image_size.1,
            ],
        )
    }

    fn create_pooling_layer(
        batch_size: usize,
        input_channel_value: usize,
        input_image_size: (usize, usize),
        info: &PoolingInformation,
    ) -> [usize; 4] {
        let output_channel_value = input_channel_value;
        let window_size = info.window_size;
        let output_image_size = (
            input_image_size.0 / window_size.0,
            input_image_size.1 / window_size.1,
        );

        // let window = Array2::zeros([window_size.0, window_size.1]);

        // let switch = Array2::from_elem([window_size.0, window_size.1], (0, 0));

        // let output = Array4::zeros([
        //     batch_size,
        //     output_channel_value,
        //     output_image_size.0,
        //     output_image_size.1,
        // ]);

        [
            batch_size,
            output_channel_value,
            output_image_size.0,
            output_image_size.1,
        ]
    }
}
