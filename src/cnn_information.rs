use std::marker::PhantomData;

pub enum LayerType {
    Convolution,
    Pooling,
    FullConnected,
    Output,
}

pub enum PoolingType {
    MaxPooling,
    AveragePooling,
}

pub enum ActivationType {
    ReLU,
}

pub enum OutputType {
    Regression,
    BinaryClassification,
    MultiClassClassification,
}

pub struct LayerInformation<T: TellLayerInformation> {
    pub layer_type: LayerType,
    pub information: T,
}

pub struct ConvolutionInInformation {
    pub filter_size: (usize, usize),
    pub stride: usize,
    pub padding: usize,
    pub activation_type: ActivationType,
}

pub struct PoolingInformation {
    pub window_size: (usize, usize),
    pub pooling_type: PoolingType,
}

pub struct FullConnectedInformation {
    pub node_value: (usize, usize),
}

pub struct OutputInformation {
    pub output_type: OutputType,
}

pub trait TellLayerInformation {}

impl TellLayerInformation for ConvolutionInInformation {}
impl TellLayerInformation for PoolingInformation {}
impl TellLayerInformation for FullConnectedInformation {}
impl TellLayerInformation for OutputInformation {}
