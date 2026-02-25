pub mod builders;
pub mod calculation_node_state;
pub mod cross_entropy_loss;
pub mod input_tensor;
pub mod linear;
pub mod nn_data_flow_node;
pub mod nn_data_flow_tree;
pub mod nn_sequential_node_flow;
pub mod relu;
pub mod softmax;
pub mod softmax_and_celoss;

use std::{
    fmt::Debug,
    ops::{Add, Div, Mul, Sub},
};

pub use cross_entropy_loss::CrossEntropyLoss;
pub use linear::Linear;
use ndarray::{ArrayD, ArrayViewD, LinalgScalar};
pub use nn_data_flow_node::NNDataFlowNodeIndexInfo;
pub use nn_data_flow_tree::NNDataFlowTree;
use num_traits::{ConstOne, ConstZero, Float};
pub use relu::ReLU;
pub use softmax::Softmax;
pub use softmax_and_celoss::SoftmaxAndCELoss;

use crate::nn_modules::input_tensor::InputTensor;

pub struct NNForwardInput<'a, 'b, T> {
    pub inputs: Vec<ArrayViewD<'a, T>>,
    pub target: Option<ArrayViewD<'b, T>>,
}

pub trait NNModule<T>: Debug {
    fn necessary_parameter_value(&self) -> usize;

    fn forward(&mut self, input: &NNForwardInput<'_, '_, T>) -> ArrayD<T>;
}

pub trait NNModuleBuilder<T> {
    type BuiltObject;
    fn new() -> Self;
    fn build(self) -> Self::BuiltObject;
}

#[derive(Debug, Clone)]
pub enum NNModuleType<T>
where
    Box<dyn NNModule<T>>: Clone,
{
    InputTensor(InputTensor<T>),
    Linear(Linear<T>),
    ReLU(ReLU<T>),
    Softmax(Softmax<T>),
    CrossEntropyLoss(CrossEntropyLoss<T>),
    SoftmaxAndCELoss(SoftmaxAndCELoss<T>),
    Custom(Box<dyn NNModule<T>>),
}

impl<T> NNModule<T> for NNModuleType<T>
where
    T: Send + Sync + LinalgScalar + Debug + ConstOne + ConstZero + PartialOrd + Float,
    for<'a> &'a T: Add<T, Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
    Box<dyn NNModule<T>>: Clone,
{
    fn necessary_parameter_value(&self) -> usize {
        match self {
            NNModuleType::InputTensor(s) => s.necessary_parameter_value(),
            NNModuleType::Linear(s) => s.necessary_parameter_value(),
            NNModuleType::ReLU(s) => s.necessary_parameter_value(),
            NNModuleType::Softmax(s) => s.necessary_parameter_value(),
            NNModuleType::CrossEntropyLoss(s) => s.necessary_parameter_value(),
            NNModuleType::SoftmaxAndCELoss(s) => s.necessary_parameter_value(),
            NNModuleType::Custom(s) => s.necessary_parameter_value(),
        }
    }

    fn forward(&mut self, input: &NNForwardInput<'_, '_, T>) -> ArrayD<T> {
        match self {
            NNModuleType::InputTensor(s) => s.forward(input),
            NNModuleType::Linear(s) => s.forward(input),
            NNModuleType::ReLU(s) => s.forward(input),
            NNModuleType::Softmax(s) => s.forward(input),
            NNModuleType::CrossEntropyLoss(s) => s.forward(input),
            NNModuleType::SoftmaxAndCELoss(s) => s.forward(input),
            NNModuleType::Custom(s) => s.forward(input),
        }
    }
}
