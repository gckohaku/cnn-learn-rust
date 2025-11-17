use ndarray::ArrayD;

pub fn im2col(inputs: &ArrayD<f64>, filters: &ArrayD<f64>, stride: usize, padding: usize) {
    // パディングとストライドについては今は考えずに、入力テンソルの次元も 4 であるとする

	if inputs.ndim() != 4 || filters.ndim() != 4 {
		panic!("次元数が4でない");
	}

	// 出力データ (行列) のサイズを計算
	// そのために、まずは入力データのそれぞれの次元のサイズを取得
	let inputs_shape = inputs.shape();
	let filters_shape = filters.shape();

	let batch_value = inputs_shape[0];
	let input_channel_value = inputs_shape[1];
	let input_size = (inputs_shape[2], inputs_shape[3]);
	let output_channel_value = filters_shape[0];
	let filter_size = (filters_shape[2], filters_shape[3]);
	let output_size = filter_size.0 * filter_size.1;

	// TODO: 行列のサイズを計算
}