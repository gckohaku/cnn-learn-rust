use crate::nn_modules::{NNModuleBuilder, Softmax};

pub struct SoftmaxBuilder {}

impl NNModuleBuilder for SoftmaxBuilder {
    type BuiltObject = Softmax;

    fn new() -> Self {
        SoftmaxBuilder {}
    }

    fn build(self) -> Self::BuiltObject {
        Softmax { output_value: None }
    }
}
