use std::{
    fmt::Debug,
    ops::{Add, Div, Mul, Sub},
};

use ndarray::{Array2, ArrayD, Ix2, LinalgScalar, ScalarOperand};
use num_traits::{ConstZero, Float};

use crate::nn_modules::{CrossEntropyLoss, NNForwardInput, NNModule, nn_data_flow_tree::NNNodeNeedsParameter, Softmax};

#[derive(Debug)]
pub struct SoftmaxAndCELoss<T> {
    pub softmax: Softmax<T>,
    pub cross_entropy_loss: CrossEntropyLoss<T>,
    pub is_grad: bool,
    // 勾配を求める時に利用
    pub(super) grad: Option<Array2<T>>,
}

impl<T> NNModule<T> for SoftmaxAndCELoss<T>
where
    T: ScalarOperand + Send + Sync + LinalgScalar + PartialOrd + Float + Debug + ConstZero,
    for<'a, 'b> &'a T: Sub<Output = T> + Div<Output = T> + Add<T, Output = T> + Mul<Output = T>,
{
    fn forward<'a, 'b>(&mut self, input: &NNForwardInput<'a, 'b, T>) -> ArrayD<T> {
        let input_values = input;

        let softmax_result = self.softmax.forward(&input_values);
        let softmax_result_view = softmax_result.view();

        let target = &input_values.target;

        if self.is_grad {
            self.grad = Some(&softmax_result.to_owned().into_dimensionality::<Ix2>().unwrap() - &target.to_owned().unwrap().into_dimensionality::<Ix2>().unwrap());
        }

        let ce_loss_input = &NNForwardInput {
            inputs: vec![softmax_result_view],
            target: Some(target.clone().unwrap()),
        };
        let loss = self.cross_entropy_loss.forward(ce_loss_input);

        let softmax_result_2d = softmax_result.into_dimensionality::<Ix2>().unwrap();
        let target_2d = input
            .target
            .to_owned()
            .unwrap()
            .into_dimensionality::<Ix2>()
            .unwrap();
        self.grad = Some(&softmax_result_2d - &target_2d);
        loss
    }
}

impl<T> NNNodeNeedsParameter for SoftmaxAndCELoss<T> {
    fn parameter_value(&self) -> usize {
        1
    }
}