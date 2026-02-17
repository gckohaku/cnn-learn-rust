use std::{fmt::Debug, ops::{Div, Sub}};

use ndarray::{Array2, ArrayD, Axis, Ix2, LinalgScalar, ScalarOperand, Zip};
use num_traits::Float;

use crate::nn_modules::{NNForwardInput, NNModule, nn_data_flow_tree::NNTreeNode};

#[derive(Debug)]
pub struct Softmax<T> {
    pub is_grad: bool,
    // 勾配を求めるとき用に出力値を保持しておく
    pub(super) grad: Option<Array2<T>>,
    pub(super) output_value: Option<Array2<T>>,
}

impl<T> NNModule<T> for Softmax<T>
where
    T: ScalarOperand + Send + Sync + LinalgScalar + PartialOrd + Float + Debug,
    for<'a> &'a T: Sub<Output = T> + Div<Output = T>
{
    fn forward<'a>(&mut self, forward_input: &NNForwardInput<T>) -> ArrayD<T> {
        let input = &forward_input.inputs[0];

        let input_2d = input.clone().into_dimensionality::<Ix2>().unwrap();
        let batch_size = input_2d.nrows();

        let max_each_sample =
            input_2d.map_axis(Axis(1), |row| row.fold(T::neg_infinity(), |m, v| if v > &m {*v} else {m}));

        // サンプルごとの最大値で引いた後に、それぞれに指数関数を適用
        let mut processed_transposed_value = Array2::zeros(input_2d.dim());

        Zip::from(&mut processed_transposed_value)
            .and(input_2d)
            .and_broadcast(&max_each_sample.to_shape((batch_size, 1)).unwrap())
            .for_each(|result, value, max| *result = (value - max).exp());

        // サンプルごとに指数関数の値の合計で除算する
        let sum_exps = processed_transposed_value.sum_axis(Axis(1));
        let mut after_softmax = Array2::zeros(processed_transposed_value.dim());

        Zip::from(&mut after_softmax)
            .and(&processed_transposed_value)
            .and_broadcast(&sum_exps.to_shape((batch_size, 1)).unwrap())
            .for_each(|result, value, sum| *result = value / sum);

        #[cfg(debug_assertions)]
        {
            dbg!(&processed_transposed_value, &after_softmax);
        }

        self.output_value = Some(after_softmax.clone());

        after_softmax.into_dyn()
    }
}

impl<T> NNTreeNode for Softmax<T> {
    const NECESSARY_VARIABLE_VALUE: usize = 1;
}