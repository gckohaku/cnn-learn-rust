use std::{cell::RefCell, fmt::Debug, rc::Rc};

use ndarray::{ArrayD, IxDyn};
use num_traits::{Float, Zero};

use crate::nn_modules::{NNDataFlowNode, NNForwardInput, NNModule};

#[derive(Debug)]
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

        Self { modules, adjacency_list, current_count }
    }

    pub fn add_module(&'a mut self, module: &'a dyn NNModule<T>) -> NNDataFlowNode<'_, T> {
        self.modules.push(Box::new(module));
        let index = self.current_count;
        self.current_count += 1;

        self.adjacency_list.push(Vec::<usize>::new());

        NNDataFlowNode::<'a, T> {
            index: index,
            tree_module: self,
        }
    }

    pub fn add_module_for_node(
        &'a mut self,
        from_index: usize,
        module: &'a dyn NNModule<T>,
    ) -> NNDataFlowNode<'a, T> {
        self.modules.push(Box::new(module));
        let to_index = self.current_count;
        self.current_count += 1;

        self.adjacency_list.push(Vec::<usize>::new());
        self.adjacency_list[from_index].push(to_index);

        NNDataFlowNode::<T> {
            index: to_index,
            tree_module: self,
        }
    }
}