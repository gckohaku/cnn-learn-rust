use std::{fmt::Debug, ops::{Add, Mul}};

use ndarray::{Array2, ArrayD, Ix2, LinalgScalar, Zip, arr0};
use num_traits::{ConstZero, Float};

use crate::nn_modules::{NNForwardInput, NNModule, nn_data_flow_tree::NNTreeNode};

#[derive(Debug)]
pub struct CrossEntropyLoss<T> {
    pub is_grad: bool,
    // 勾配を求めるために期待値を保持しておく
    pub(super) expected_value: Option<Array2<T>>,
}

impl<T> NNModule<T> for CrossEntropyLoss<T>
where
    T: Send + Sync + LinalgScalar + Float + ConstZero + Debug,
    for<'a> &'a T: Add<T, Output = T> + Mul<Output = T>
{
    fn forward<'a>(&mut self, input: &NNForwardInput<'_, '_, T>) -> ArrayD<T> {
        let nn_result = &input.inputs[0];
        let target = input.target.as_ref().unwrap();

        let nn_result_2d = nn_result.clone().into_dimensionality::<Ix2>().unwrap();
        let target_2d = target.clone().into_dimensionality::<Ix2>().unwrap();

        let ln_output = nn_result_2d.map(|x| (x + T::epsilon()).ln());

        let error = -Zip::from(target_2d)
            .and(&ln_output)
            .fold(T::ZERO, |t, e, o| t + e * o);

        arr0(error).into_dyn()
    }
}

impl<T> NNTreeNode for CrossEntropyLoss<T> {
    const NECESSARY_VARIABLE_VALUE: usize = 1;
}