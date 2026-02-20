use ndarray::ArrayD;

use crate::nn_modules::NNModule;

// これは計算を行う NNModule の集合体と入力パラメータのインターフェースの役割を担う
// NNModule として扱うことで他の NNModule と同じように扱うことができる
#[derive(Debug)]
pub struct InputTensor {}

impl<T> NNModule<T> for InputTensor
where
    T: Clone + std::fmt::Debug,
{
    fn necessary_parameter_value(&self) -> usize {
        0
    }

    fn forward(&mut self, input: &super::NNForwardInput<'_, '_, T>) -> ArrayD<T> {
        return input.inputs[0].to_owned();
    }
}

impl InputTensor {}
