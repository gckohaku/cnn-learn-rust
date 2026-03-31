use ndarray::{Array2, Array4, Array6, Slice, s};
use ndarray_ndimage::{PadMode, pad};
use num_traits::{Float, FromPrimitive, Num};

use crate::nn_modules::NNNecessaryTraits;

pub fn im2col<T>(
    inputs: &Array4<T>,
    filters: &Array4<T>,
    stride: usize,
    padding: usize,
) -> (Array2<T>, Array2<T>)
where
    T: Clone + Copy + Send + Sync + Num + Float + FromPrimitive,
{
    let padded_inputs = pad(
        inputs,
        &[[0, 0], [0, 0], [padding, padding], [padding, padding]],
        PadMode::Constant(T::from(0.0).expect("Cast to T from float is failed.")),
    );

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
    // let output_matrix_size = (
    //     output_channel_value,
    //     batch_value * output_size.0 * output_size.1,
    // );

    let mut processing_tensor: Array6<T> = Array6::zeros((
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
                        Slice::from(h..(h + output_size.0 * stride).min(inputs_shape[2]))
                            .step_by(stride.try_into().unwrap()),
                        Slice::from(w..(w + output_size.1 * stride).min(inputs_shape[3]))
                            .step_by(stride.try_into().unwrap())
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

pub fn im2col_for_pooling<T>(
    inputs: &Array4<T>,
    stride: usize,
    window_size: (usize, usize),
) -> Array2<T>
where
    T: NNNecessaryTraits,
{
    // 出力データ (行列) のサイズを計算
    // そのために、まずは入力データのそれぞれの次元のサイズを取得
    let inputs_shape = inputs.shape();

    let batch_value = inputs_shape[0];
    let input_channel_value = inputs_shape[1];
    let input_size = (inputs_shape[2], inputs_shape[3]);
    // let output_channel_value = input_channel_value;
    let output_size = (
        (input_size.0 - window_size.0) / stride + 1,
        (input_size.1 - window_size.1) / stride + 1,
    );

    let mut processing_tensor: Array6<T> = Array6::zeros((
        batch_value,
        input_channel_value,
        window_size.0,
        window_size.1,
        output_size.0,
        output_size.1,
    ));

    for h in 0..window_size.0 {
        for w in 0..window_size.1 {
            processing_tensor
                .slice_mut(s![.., .., h, w, .., ..])
                .assign(&inputs.slice(s![
                        ..,
                        ..,
                        Slice::from(h..(h + output_size.0 * stride).min(inputs_shape[2]))
                            .step_by(stride.try_into().unwrap()),
                        Slice::from(w..(w + output_size.1 * stride).min(inputs_shape[3]))
                            .step_by(stride.try_into().unwrap())
                    ]));
        }
    }

    processing_tensor.swap_axes(0, 2);
    processing_tensor.swap_axes(1, 3);

    let input_col_matrix = processing_tensor
        .to_shape((
            window_size.0 * window_size.1,
            batch_value * input_channel_value * output_size.0 * output_size.1,
        ))
        .unwrap();

    input_col_matrix.to_owned()
}
