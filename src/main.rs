mod cnn_activations;
mod cnn_information;
mod cnn_network;
mod cnn_transformations;
mod convolution_network;
mod fully_connected_network;
mod inspection_cnn;
mod mnist_test;
mod mnist_test_cnn;
mod nn_modules;
mod rand;
mod simple_dataset;
mod test_cnn_network;
mod test_nn_modules;
mod type_utilities;
mod utilities;
mod lr_range_test;
mod learning_rate_manager;

fn main() {
    let _ = mnist_test_cnn::mnist_process();
    // let _ = lr_range_test::test();
}
