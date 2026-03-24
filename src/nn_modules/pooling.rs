use ndarray::{Array1, Array2, ArrayD, Axis, Ix4, Zip};

use crate::{
    cnn_information::PoolingType,
    cnn_transformations,
    nn_modules::{NNModule, NNNecessaryTraits},
};

#[derive(Debug)]
pub struct Pooling<T> {
    pooling_type: PoolingType,
    window: Array2<T>,
    window_size: (usize, usize),
    stride: usize,
    pooling_mask: Array1<usize>,
}

impl<T> NNModule<T> for Pooling<T>
where
    T: NNNecessaryTraits,
{
    fn necessary_parameter_value(&self) -> usize {
        1
    }

    fn forward(
        &mut self,
        input: &super::NNForwardInput<'_, '_, T>,
        is_grad: bool,
    ) -> ndarray::ArrayD<T> {
        let mut result_tensor = ArrayD::<T>::zeros(vec![]);

        let input_4d = input.inputs[0]
            .clone()
            .into_dimensionality::<Ix4>()
            .unwrap();
        let input_shape = input_4d.shape();

        // max pooling 以外を用いる時は、match で分岐させ、それぞれの処理も関数に分けた方が良さそう
        if self.pooling_type == PoolingType::MaxPooling {
            // やっていること自体は平坦化してベクトル処理がしやすい形にしてから max pooling
            // 処理が終わったら形状を戻すことも忘れずに
            let spread_input = cnn_transformations::im2col_for_pooling(
                &input_4d.to_owned(),
                self.stride,
                self.window_size,
            );

            let window_value = self.window_size.0 * self.window_size.1;
            let spread_cols_value =
                input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3] / window_value;
            let mut mask = Array1::zeros(spread_cols_value);
            let mut after_pooling = Array1::zeros(spread_cols_value);

            Zip::from(after_pooling.view_mut())
                .and(mask.view_mut())
                .and(spread_input.axis_iter(Axis(1)))
                .par_for_each(|res, mask, col| {
                    let mut max_value = T::neg_infinity();
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

            self.pooling_mask = mask.to_owned();

            let reshape_pooling = after_pooling
                .to_shape((
                    input_shape[0],
                    input_shape[1],
                    input_shape[2] / self.window_size.0,
                    input_shape[3] / self.window_size.1,
                ))
                .unwrap();

            result_tensor = reshape_pooling.into_dyn().to_owned();
        }

        result_tensor
    }

    fn propagate_grad(
        &mut self,
        grad: Option<&ndarray::ArrayViewD<T>>,
        eta: T,
    ) -> ndarray::ArrayD<T> {
        let mut result_tensor = ArrayD::<T>::zeros(vec![]);

        // こっちも必要になったら match に変えて関数分けしたい
        if self.pooling_type == PoolingType::MaxPooling {
            // こっちでも平坦化して処理を行う
            // let grad_4d = grad
            //     .unwrap()
            //     .to_owned()
            //     .into_dimensionality::<Ix4>()
            //     .unwrap();

            let pooling_ncols = self.pooling_mask.len();

            let temporary_owned_grad = grad.unwrap().to_owned();
            let vectored_pooled = temporary_owned_grad.to_shape(pooling_ncols).unwrap();

            let window_size = self.window_size;
            let pooling_nrows = window_size.0 * window_size.1;
            let mut spread_grad_for_before = Array2::<T>::zeros((pooling_ncols, pooling_nrows));

            // mask が指すインデックスにそれぞれの勾配を渡す
            Zip::from(spread_grad_for_before.axis_iter_mut(Axis(0)))
                .and(&vectored_pooled)
                .and(&self.pooling_mask)
                .par_for_each(|mut v, d, m| v[*m] = *d);

			result_tensor = spread_grad_for_before.into_dyn().to_owned();
        }

        result_tensor
    }
}
