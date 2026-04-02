use std::marker::PhantomData;

use ndarray::ArrayD;

use crate::{impl_as_any_with_mut, nn_modules::{NNModule, NNNecessaryTraits}};

// これは計算を行う NNModule の集合体と入力パラメータのインターフェースの役割を担う
// NNModule として扱うことで他の NNModule と同じように扱うことができる
#[derive(Debug, Clone)]
pub struct InputTensor<T> {
    pub phantom: PhantomData<T>,
}

impl<T> NNModule<T> for InputTensor<T>
where
    T: NNNecessaryTraits,
{
    fn necessary_parameter_value(&self) -> usize {
        1
    }

    fn forward(&mut self, input: &super::NNForwardInput<'_, '_, T>, _is_grad: bool) -> ArrayD<T> {
        return input.inputs[0].to_owned();
    }

    fn propagate_grad(&mut self, grad: Option<&ndarray::ArrayViewD<T>>, _eta: T) -> ArrayD<T> {
        // 入力の勾配をそのまま返す
        grad.unwrap().to_owned()
    }

    impl_as_any_with_mut!();
}

impl<T> InputTensor<T> {}
