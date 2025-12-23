use crate::{
    cnn_information::{
        ActivationType, ConvolutionInformation, LayerInformation, LayerInformationContent,
        LayerType, OutputInformation, OutputType, PoolingInformation, PoolingType,
    },
    cnn_network::NeuralNetworkCNN,
};

pub fn inspection() {
    let data = ndarray::array![[[
        [0.25, 1.0, 0.75, 0.0],
        [0.5, 0.75, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 0.25, 0.5, 0.75]
    ]]];
    let expect = ndarray::array![[0.0, 1.0, 0.0]];

    let mut nn = make_neural_network();

    dbg!(&data);
    println!("forward");
    nn.forward(&data, &expect);
    println!("error");
    dbg!(nn.get_error());
    println!("backward");
    nn.backward(&expect, 0.01);
}

fn make_neural_network() -> NeuralNetworkCNN {
    NeuralNetworkCNN::new(
        1,
        1,
        (4, 4),
        vec![3, 3],
        vec![
            LayerInformation {
                layer_type: LayerType::Convolution,
                information: LayerInformationContent::Convolution(ConvolutionInformation {
                    filter_size: (2, 2),
                    filter_value: 3,
                    stride: 2,
                    padding: 0,
                    activation_type: ActivationType::ReLU,
                }),
            },
            LayerInformation {
                layer_type: LayerType::Pooling,
                information: LayerInformationContent::Pooling(PoolingInformation {
                    window_size: (2, 2),
                    pooling_type: PoolingType::MaxPooling,
                    stride: 2,
                }),
            },
        ],
        OutputInformation {
            node_value: 3,
            output_type: OutputType::MultiClassClassification,
        },
    )
}
