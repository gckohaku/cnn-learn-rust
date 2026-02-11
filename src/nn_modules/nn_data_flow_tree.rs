use crate::nn_modules::{NNDataFlowNode, NNDataFlowNodeIndexInfo, NNModule};
use std::fmt::Debug;

#[derive(Debug, Clone)]
pub struct NNDataFlowTree<'a, T> {
    pub modules: Vec<Box<&'a dyn NNModule<T>>>,
    adjacency_list: Vec<Vec<usize>>,
    current_count: usize,
}

// impl<'a, T> NNModule<T> for NNDataFlowTree<T>
// where
//     T: Clone + Zero,
// {
//     fn forward(&mut self, input: &NNForwardInput<'_, '_, T>) -> ArrayD<T> {

//     }
// }

impl<'a, T> NNDataFlowTree<'a, T> {
    pub fn new() -> Self {
        let modules = Vec::<Box<&'_ dyn NNModule<T>>>::new();
        let adjacency_list = Vec::<Vec<usize>>::new();
        let current_count = 0;

        Self {
            modules,
            adjacency_list,
            current_count,
        }
    }

    pub fn add_from_root(&mut self, module: &'a dyn NNModule<T>) -> NNDataFlowNodeIndexInfo {
        self.modules.push(Box::new(module));
        let index = self.current_count;
        self.current_count += 1;

        self.adjacency_list.push(Vec::<usize>::new());

        NNDataFlowNodeIndexInfo { index }
    }

    pub fn add(
        &mut self,
        from: &NNDataFlowNodeIndexInfo,
        module: &'a dyn NNModule<T>,
    ) -> NNDataFlowNodeIndexInfo {
        self.modules.push(Box::new(module));
        let from_index = from.index;
        let to_index = self.current_count;
        self.current_count += 1;

        self.adjacency_list.push(Vec::<usize>::new());
        self.adjacency_list[from_index].push(to_index);

        NNDataFlowNodeIndexInfo { index: from_index }
    }
}
