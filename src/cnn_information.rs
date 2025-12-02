use std::marker::PhantomData;

use crate::{makeHList, type_utilities::{Here, Member}};

#[derive(Debug, Clone, PartialEq)]
pub enum LayerType {
    Convolution,
    Pooling,
    FullConnected,
    Output,
}

#[derive(Debug, Clone, Copy)]
pub enum PoolingType {
    MaxPooling,
    AveragePooling,
}

#[derive(Debug, Clone, Copy)]
pub enum ActivationType {
    ReLU,
}

#[derive(Debug, Clone, Copy)]
pub enum OutputType {
    Regression,
    BinaryClassification,
    MultiClassClassification,
}

#[derive(Debug, Clone, Copy)]
pub enum LayerInformationContent {
    Convolution(ConvolutionInformation),
    Pooling(PoolingInformation),
    FullConnected(FullConnectedInformation),
    Output(OutputInformation),
}

impl LayerInformationContent {
    pub fn as_convolution(&self) -> Option<&ConvolutionInformation> {
        match self {
            LayerInformationContent::Convolution(c) => Some(c),
            _ => None,
        }
    }

    pub fn as_pooling(&self) -> Option<&PoolingInformation> {
        match self {
            LayerInformationContent::Pooling(p) => Some(p),
            _ => None,
        }
    }

    pub fn as_full_connected(&self) -> Option<&FullConnectedInformation> {
        match self {
            LayerInformationContent::FullConnected(f) => Some(f),
            _ => None,
        }
    }

    pub fn as_output(&self) -> Option<&OutputInformation> {
        match self {
            LayerInformationContent::Output(o) => Some(o),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LayerInformation {
    pub layer_type: LayerType,
    pub information: LayerInformationContent,
}

#[derive(Debug, Clone, Copy)]
pub struct ConvolutionInformation {
    pub filter_size: (usize, usize),
    pub filter_value: usize,
    pub stride: usize,
    pub padding: usize,
    pub activation_type: ActivationType,
}

#[derive(Debug, Clone, Copy)]
pub struct PoolingInformation {
    pub window_size: (usize, usize),
    pub pooling_type: PoolingType,
}

#[derive(Debug,Clone,  Copy)]
pub struct FullConnectedInformation {
    pub node_value: usize,
    pub activation_type: ActivationType,
}

#[derive(Debug,Clone,  Copy)]
pub struct OutputInformation {
    pub node_value: usize,
    pub output_type: OutputType,
}

type InformationContentList = makeHList!(ConvolutionInformation, PoolingInformation, FullConnectedInformation, OutputInformation);