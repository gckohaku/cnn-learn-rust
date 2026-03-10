use std::fmt::Debug;

use ndarray::{ArrayD, Zip, parallel::prelude::*};
use num_traits::{ConstOne, ConstZero, Float, FloatConst, Num};

use crate::nn_modules::{NNForwardInput, NNModule, NNNecessaryTraits};

#[derive(Debug, Clone)]
pub struct ReLU<T> {
    pub is_grad: bool,
    // 勾配計算のために保持するデータ
    pub(super) output_value: Option<ArrayD<T>>,
    pub(super) grad: Option<ArrayD<T>>,
}

impl<T> NNModule<T> for ReLU<T>
where
    T: NNNecessaryTraits,
{
    fn necessary_parameter_value(&self) -> usize {
        1
    }

    fn forward<'a>(&mut self, forward_input: &NNForwardInput<'_, '_, T>) -> ArrayD<T> {
        let input = &forward_input.inputs[0];
        let mut clone_array = input.to_owned();
        clone_array.par_mapv_inplace(|x| if x > T::ZERO { x } else { T::ZERO });
        self.output_value = Some(clone_array.clone());
        if self.is_grad {
            self.grad = Some(self.calc_grad(&clone_array));
        }
        return clone_array;
    }

    fn propagate_grad(&mut self, grad: Option<&ndarray::ArrayViewD<T>>, eta: T) -> ArrayD<T> {
        let mut before_grad = grad.unwrap().to_owned();
        Zip::from(&before_grad).and(&self.grad.clone().unwrap()).par_for_each(|b, s: &T| *b *= *s);
        before_grad
    }
}

impl<T> ReLU<T>
where
    T: Send + Sync + Num + Float + ConstOne + ConstZero,
{
    fn calc_grad(&mut self, result: &ArrayD<T>) -> ArrayD<T> {
        let mut grad = result.clone();
        grad.par_mapv_inplace(|y: T| if y > T::ZERO { T::ONE } else { T::ZERO });
        grad
    }
}
