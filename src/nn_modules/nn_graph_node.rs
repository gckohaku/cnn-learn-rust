use crate::nn_modules::NNModule;

// ループごとに新しくインスタンスを生成するか、Vec<Arc<T>> にすることを検討した方がいいかも
pub struct NNGraphNode<'a, T> {
	pub module_data: &'a dyn NNModule<T>,
	pub child_nodes: Vec<NNGraphNode<'a, T>>,
}