pub mod batch_norm_2d;
pub mod builders;
pub mod calculation_node_state;
pub mod convolution;
pub mod cross_entropy_loss;
pub mod input_tensor;
pub mod linear;
pub mod max_pooling;
pub mod nn_data_flow_node;
pub mod nn_data_flow_tree;
pub mod nn_sequential_node_flow;
pub mod relu;
pub mod reshape_tensor;
pub mod softmax;
pub mod softmax_and_celoss;

use std::{any::Any, fmt::Debug};

pub use batch_norm_2d::BatchNorm2d;
pub use convolution::Convolution;
pub use cross_entropy_loss::CrossEntropyLoss;
use dyn_clone::{DynClone, clone_trait_object};
pub use linear::Linear;
pub use max_pooling::MaxPooling;
use ndarray::{ArrayD, ArrayViewD, ScalarOperand};
pub use nn_data_flow_node::NNDataFlowNodeIndexInfo;
pub use nn_data_flow_tree::NNDataFlowTree;
use num_traits::{ConstOne, ConstZero, Float, FromPrimitive, NumAssign, One, Zero};
pub use relu::ReLU;
pub use reshape_tensor::ReshapeTensor;
pub use softmax::Softmax;
pub use softmax_and_celoss::SoftmaxAndCELoss;

use crate::nn_modules::input_tensor::InputTensor;

pub struct NNForwardInput<'a, 'b, T> {
    pub inputs: Vec<ArrayViewD<'a, T>>,
    pub target: Option<ArrayViewD<'b, T>>,
}

pub trait NNModule<T>: Debug + DynClone + Send + Sync {
    fn necessary_parameter_value(&self) -> usize;

    fn forward(
        &mut self,
        input: &NNForwardInput<'_, '_, T>,
        is_grad: bool,
    ) -> Result<ArrayD<T>, &'static str>;
    // 逆伝播処理
    fn propagate_grad(&mut self, grad: Option<&ArrayViewD<T>>, eta: T) -> ArrayD<T>;

    // 抽象型から具体型に変換する際に必要
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

clone_trait_object!(<T> NNModule<T>);

#[macro_export]
macro_rules! impl_as_any_with_mut {
    () => {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
    };
}

pub trait NNModuleBuilder<T> {
    type BuiltObject;
    fn new() -> Self;
    fn build(self) -> Self::BuiltObject;
}

pub trait NNNecessaryTraits:
    Send
    + Sync
    + Debug
    + Float
    + One
    + ConstOne
    + Zero
    + ConstZero
    + NumAssign
    + FromPrimitive
    + ScalarOperand
    + 'static
{
}

impl<T> NNNecessaryTraits for T where
    T: Send
        + Sync
        + Debug
        + Float
        + One
        + ConstOne
        + Zero
        + ConstZero
        + NumAssign
        + FromPrimitive
        + ScalarOperand
        + 'static
{
}
