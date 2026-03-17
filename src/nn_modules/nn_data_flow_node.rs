// #[derive(Debug)]
// pub struct NNDataFlowNode<'a, T> {
// 	pub index: usize,
// 	pub add_module_callback: fn (usize, &'a dyn NNModule<T>) -> NNDataFlowNode<'_, T>,
// }


// impl<'a, T> NNDataFlowNode<'a, T> {
// 	pub fn add_module(&'a mut self, module: &'a dyn NNModule<T>) -> NNDataFlowNode<'a, T> {
// 		(self.add_module_callback)(self.index, module)
// 	}
// }

#[derive(Debug)]
pub struct NNDataFlowNodeIndexInfo {
	pub index: usize,
}