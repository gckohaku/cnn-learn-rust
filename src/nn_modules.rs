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
pub mod convolution;
pub mod pooling;
pub mod reshape_tensor;

use std::fmt::Debug;

pub use cross_entropy_loss::CrossEntropyLoss;
pub use linear::Linear;
use ndarray::{ArrayD, ArrayViewD};
pub use nn_data_flow_node::NNDataFlowNodeIndexInfo;
pub use nn_data_flow_tree::NNDataFlowTree;
use num_traits::{ConstOne, ConstZero, Float, FromPrimitive, NumAssign};
pub use relu::ReLU;
pub use softmax::Softmax;
pub use softmax_and_celoss::SoftmaxAndCELoss;
pub use convolution::Convolution;
pub use pooling::Pooling;
pub use reshape_tensor::ReshapeTensor;

use crate::nn_modules::input_tensor::InputTensor;

pub struct NNForwardInput<'a, 'b, T> {
    pub inputs: Vec<ArrayViewD<'a, T>>,
    pub target: Option<ArrayViewD<'b, T>>,
}

pub trait NNModule<T>: Debug {
    fn necessary_parameter_value(&self) -> usize;

    fn forward(&mut self, input: &NNForwardInput<'_, '_, T>, is_grad: bool) -> ArrayD<T>;
    // 逆伝播処理
    fn propagate_grad(&mut self, grad: Option<&ArrayViewD<T>>, eta: T) -> ArrayD<T>;
}

pub trait NNModuleBuilder<T> {
    type BuiltObject;
    fn new() -> Self;
    fn build(self) -> Self::BuiltObject;
}

#[derive(Debug, Clone)]
pub enum NNModuleType<T> where T: NNNecessaryTraits {
    InputTensor(InputTensor<T>),
    Linear(Linear<T>),
    ReLU(ReLU<T>),
    Softmax(Softmax<T>),
    CrossEntropyLoss(CrossEntropyLoss<T>),
    SoftmaxAndCELoss(SoftmaxAndCELoss<T>),
    // Custom(Box<dyn NNModule<T>>),
}

pub trait NNNecessaryTraits: Send + Sync + Debug + Float + ConstOne + ConstZero + NumAssign + FromPrimitive + 'static {}

impl<T> NNNecessaryTraits for T where T: Send + Sync + Debug + Float + ConstOne + ConstZero + NumAssign + FromPrimitive + 'static
{}

impl<T> NNModule<T> for NNModuleType<T>
where
    T: NNNecessaryTraits,
{
    fn necessary_parameter_value(&self) -> usize {
        match self {
            NNModuleType::InputTensor(s) => s.necessary_parameter_value(),
            NNModuleType::Linear(s) => s.necessary_parameter_value(),
            NNModuleType::ReLU(s) => s.necessary_parameter_value(),
            NNModuleType::Softmax(s) => s.necessary_parameter_value(),
            NNModuleType::CrossEntropyLoss(s) => s.necessary_parameter_value(),
            NNModuleType::SoftmaxAndCELoss(s) => s.necessary_parameter_value(),
            // NNModuleType::Custom(s) => s.necessary_parameter_value(),
        }
    }

    fn forward(&mut self, input: &NNForwardInput<'_, '_, T>, is_grad: bool) -> ArrayD<T> {
        match self {
            NNModuleType::InputTensor(s) => s.forward(input, is_grad),
            NNModuleType::Linear(s) => s.forward(input, is_grad),
            NNModuleType::ReLU(s) => s.forward(input, is_grad),
            NNModuleType::Softmax(s) => s.forward(input, is_grad),
            NNModuleType::CrossEntropyLoss(s) => s.forward(input, is_grad),
            NNModuleType::SoftmaxAndCELoss(s) => s.forward(input, is_grad),
            // NNModuleType::Custom(s) => s.forward(input),
        }
    }

    fn propagate_grad(&mut self, grad: Option<&ArrayViewD<T>>, eta: T) -> ArrayD<T> {
        match self {
            NNModuleType::InputTensor(s) => s.propagate_grad(grad, eta),
            NNModuleType::Linear(s) => s.propagate_grad(grad, eta),
            NNModuleType::ReLU(s) => s.propagate_grad(grad, eta),
            NNModuleType::Softmax(s) => s.propagate_grad(grad, eta),
            NNModuleType::CrossEntropyLoss(s) => s.propagate_grad(grad, eta),
            NNModuleType::SoftmaxAndCELoss(s) => s.propagate_grad(grad, eta),
            // NNModuleType::Custom(s) => s.forward(input),
        }
    }
}
