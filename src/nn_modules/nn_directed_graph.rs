use ndarray::ArrayD;

use crate::nn_modules::{NNForwardInput, NNGraphNode, NNModule};

pub struct NNDirectedGraph<'a, T> {
    pub graph: NNGraphNode<'a, T>,
}

// impl<'a, T> NNModule<T> for NNDirectedGraph<'a, T> {
//     fn forward(&mut self, input: &NNForwardInput<'_, '_, T>) -> ArrayD<T> {
//         let parent_data = self.graph.module_data.forward(input);
//     }
// }

impl<'a, T> NNDirectedGraph<'a, T> {}

// fn recursive_forward<'a, T>(
//     &mut self,
//     input: &NNForwardInput<'_, '_, T>,
//     node: &NNGraphNode<'a, T>,
// ) -> ArrayD<T> {
//     let parent_data = self.graph.module_data.forward(input);

//     let next_input = NNForwardInput::<T> {
//         input: parent_data.view(),
//         target: input.target.to_owned(),
//     };

// 	let child

//     for child in node.child_nodes {
// 		recursive_forward(&mut self, input, node)
// 	}
// }
