use std::marker::PhantomData;

use crate::{makeHList, type_utilities::{Here, Member}};

#[derive(Debug, PartialEq)]
pub enum LayerType {
    Convolution,
    Pooling,
    FullConnected,
    Output,
}

#[derive(Debug)]
pub enum PoolingType {
    MaxPooling,
    AveragePooling,
}

#[derive(Debug)]
pub enum ActivationType {
    ReLU,
}

#[derive(Debug)]
pub enum OutputType {
    Regression,
    BinaryClassification,
    MultiClassClassification,
}

#[derive(Debug)]
pub enum LayerInformationContent {
    Convolution(ConvolutionInInformation),
    Pooling(PoolingInformation),
    FullConnected(FullConnectedInformation),
    Output(OutputInformation),
}

#[derive(Debug)]
pub struct LayerInformation {
    pub layer_type: LayerType,
    pub information: LayerInformationContent,
}

#[derive(Debug)]
pub struct ConvolutionInInformation {
    pub filter_size: (usize, usize),
    pub filter_value: usize,
    pub stride: usize,
    pub padding: usize,
    pub activation_type: ActivationType,
}

#[derive(Debug)]
pub struct PoolingInformation {
    pub window_size: (usize, usize),
    pub pooling_type: PoolingType,
}

#[derive(Debug)]
pub struct FullConnectedInformation {
    pub node_value: usize,
    pub activation_type: ActivationType,
}

#[derive(Debug)]
pub struct OutputInformation {
    pub node_value: usize,
    pub output_type: OutputType,
}
