use std::fs;

use ndarray::{Array2, Array4, Array6, ArrayD, Shape, Slice, s};
use ndarray_ndimage::{PadMode, pad};

pub fn im2col(
    inputs: &Array4<f64>,
    filters: &Array4<f64>,
    stride: usize,
    padding: usize,
) -> (Array2<f64>, Array2<f64>) {
    if stride <= 0 {
        println!("{}, {}", stride, stride <= 0);
        panic!("'stride' must be greater than zero.");
    }

    if inputs.ndim() != 4 || filters.ndim() != 4 {
        panic!("次元数が4でない");
    }

    let padded_inputs = pad(inputs, &[[0, 0], [0, 0], [padding, padding], [padding, padding]], PadMode::Constant(0.0));

    // 出力データ (行列) のサイズを計算
    // そのために、まずは入力データのそれぞれの次元のサイズを取得
    let inputs_shape = padded_inputs.shape();
    let filters_shape = filters.shape();

    let batch_value = inputs_shape[0];
    let input_channel_value = inputs_shape[1];
    let input_size = (inputs_shape[2], inputs_shape[3]);
    let output_channel_value = filters_shape[0];
    let filter_size = (filters_shape[2], filters_shape[3]);
    let output_size = (
        (input_size.0 - filter_size.0) / stride + 1,
        (input_size.1 - filter_size.1) / stride + 1,
    );

    // 行列のサイズを計算
    let output_matrix_size = (
        output_channel_value,
        batch_value * output_size.0 * output_size.1,
    );

    let mut processing_tensor: Array6<f64> = Array6::zeros((
        batch_value,
        input_channel_value,
        filter_size.0,
        filter_size.1,
        output_size.0,
        output_size.1,
    ));

    for h in 0..filter_size.0 {
        for w in 0..filter_size.1 {
            processing_tensor
                .slice_mut(s![.., .., h, w, .., ..])
                .assign(&padded_inputs.slice(s![
                    ..,
                    ..,
                    Slice::from(h..(h + output_size.0 * stride).min(inputs_shape[2])).step_by(stride.try_into().unwrap()),
                    Slice::from(w..(w + output_size.1 * stride).min(inputs_shape[3])).step_by(stride.try_into().unwrap())
                ]));
        }
    }

    processing_tensor.swap_axes(0, 1);
    processing_tensor.swap_axes(1, 2);
    processing_tensor.swap_axes(2, 3);

    let input_col_matrix = processing_tensor
        .to_shape((
            input_channel_value * filter_size.0 * filter_size.1,
            batch_value * output_size.0 * output_size.1,
        ))
        .unwrap();

    let filter_col_matrix = filters
        .to_shape((
            output_channel_value,
            input_channel_value * filter_size.0 * filter_size.1,
        ))
        .unwrap();

    (input_col_matrix.to_owned(), filter_col_matrix.to_owned())
}
