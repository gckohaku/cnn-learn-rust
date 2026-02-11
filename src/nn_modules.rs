pub mod builders;
pub mod cross_entropy_loss;
pub mod linear;
pub mod relu;
pub mod softmax;
pub mod softmax_and_celoss;
pub mod nn_data_flow_node;
pub mod nn_data_flow_tree;

use std::fmt::Debug;

pub use cross_entropy_loss::CrossEntropyLoss;
pub use linear::Linear;
use ndarray::{ArrayD, ArrayViewD};
use num_traits::Zero;
pub use relu::ReLU;
pub use softmax::Softmax;
pub use softmax_and_celoss::SoftmaxAndCELoss;
pub use nn_data_flow_tree::NNDataFlowTree;
pub use nn_data_flow_node::NNDataFlowNode;
pub use nn_data_flow_node::NNDataFlowNodeIndexInfo;

pub struct NNForwardInput<'a, 'b, T> {
    pub input: ArrayViewD<'a, T>,
    pub target: Option<ArrayViewD<'b, T>>,
}

pub trait NNModule<T>: Debug {
    fn forward(&mut self, input: &NNForwardInput<'_, '_, T>) -> ArrayD<T>;
}

pub trait NNModuleBuilder<T>
where
{
    type BuiltObject;
    fn new() -> Self;
    fn build(self) -> Self::BuiltObject;
}
