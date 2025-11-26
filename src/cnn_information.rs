use std::marker::PhantomData;

use crate::{makeHList, type_utilities::{Here, Member}};

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

// pub struct LayerInformationUnion<T>
// where T: Member<InformationList, Here>{
//     pub layer_type: LayerType,
//     pub information: T,
// }

pub enum LayerInformation {
    ConvolutionInInformation(ConvolutionInInformation),
    
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

pub type InformationList = makeHList!(ConvolutionInInformation, PoolingInformation, FullConnectedInformation, OutputInformation);

// pub trait TellLayerInformation {}

// impl TellLayerInformation for ConvolutionInInformation {}
// impl TellLayerInformation for PoolingInformation {}
// impl TellLayerInformation for FullConnectedInformation {}
// impl TellLayerInformation for OutputInformation {}
