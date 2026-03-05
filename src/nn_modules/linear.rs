use std::fmt::Debug;

use ndarray::{Array1, Array2, ArrayD, Ix2, LinalgScalar};
use num_traits::{AsPrimitive, Float, Num, PrimInt};

use crate::nn_modules::{NNForwardInput, NNModule, NNNecessaryTraits};

/// アフィン変換を行うニューラルネットワークモジュール
#[derive(Clone)]
pub struct Linear<T> {
    pub weights: Array2<T>,
    pub biases: Array1<T>,
    pub is_grad: bool,
    // 重みを更新するために保持するデータ
    pub(super) input_value: Option<Array2<T>>,
    pub(super) grad: Option<Array2<T>>,
}

impl<T> Linear<T> {
    #[cfg(debug_assertions)]
    pub fn debug_set_weights(&mut self, w: Array2<T>) {
        self.weights = w;
    }

    #[cfg(debug_assertions)]
    pub fn debug_set_biases(&mut self, b: Array1<T>) {
        self.biases = b;
    }
}

impl<T> NNModule<T> for Linear<T>
where
    T: NNNecessaryTraits,
{
    fn necessary_parameter_value(&self) -> usize {
        1
    }

    fn forward(&mut self, forward_input: &NNForwardInput<'_, '_, T>) -> ArrayD<T> {
        let input = &forward_input.inputs[0];
        let input_2d = input.clone().into_dimensionality::<Ix2>().unwrap();

        // Linear 層では、勾配計算のために入力行列の転置を保持しておけばよい
        if self.is_grad == true {
            self.grad = Some(input_2d.t().to_owned());
        }

        let result = input_2d.dot(&self.weights) + &self.biases;
        result.into_dyn()
    }
}

impl<T> Debug for Linear<T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Linear {{")?;
        writeln!(f, "    weights shape: {:?}", self.weights.shape())?;
        writeln!(f, "    biases shape: {:?}", self.biases.shape())?;
        writeln!(f, "    is_grad: {:?}", self.is_grad)?;
        writeln!(f, "    grad: {:?}", self.grad)?;
        writeln!(f, "}}")?;
        Ok(())
    }
}

