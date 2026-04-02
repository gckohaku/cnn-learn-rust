use ndarray::ArrayD;
use num_traits::{ConstOne, ConstZero, Float};

use crate::{impl_as_any_with_mut, nn_modules::{
    NNDataFlowNodeIndexInfo, NNForwardInput, NNModule, NNNecessaryTraits,
    calculation_node_state::CalculationNodeState, input_tensor::InputTensor,
    nn_sequential_node_flow::NNSequentialNodeFlow,
}};
use std::{collections::VecDeque, fmt::Debug, marker::PhantomData, usize};

#[derive(Debug, Clone)]
pub struct NNDataFlowTree<T>
where
    T: NNNecessaryTraits,
{
    pub modules: Vec<Box<dyn NNModule<T>>>,
    adjacency_list: Vec<Vec<usize>>,
    inverse_adjacency_list: Vec<Vec<usize>>,
    current_count: usize,
    root_node_parameters: Vec<usize>,
    // 計算結果を使用する回数　この値が 0 になるまでは clone する
    // use_calculation_result_times: Vec<usize>,
    variables_stocks: Vec<Vec<ArrayD<T>>>,
    // input_variables_indices: Vec<Vec<usize>>,
    sequential_flow: NNSequentialNodeFlow,
    inverse_sequential_flow: NNSequentialNodeFlow,
    is_not_yet_executed: bool,
}

impl<'a, T> NNModule<T> for NNDataFlowTree<T>
where
    T: NNNecessaryTraits,
{
    fn necessary_parameter_value(&self) -> usize {
        usize::MAX
    }

    fn forward(&mut self, input: &NNForwardInput<'_, '_, T>, is_grad: bool) -> ArrayD<T> {
        if self.is_not_yet_executed {
            self.calc_sequential_node_flow();
            self.is_not_yet_executed = false;
        }

        let target = input.target.as_ref().unwrap();
        let mut loop_result = ArrayD::<T>::zeros(vec![]);

        // まず、すべての入力引数を適切なところに保管する
        self.variables_stocks.resize(self.modules.len(), vec![]);
        for i in 0..self.root_node_parameters.len() {
            let set_index = self.root_node_parameters[i];
            self.variables_stocks[set_index].push(input.inputs[i].to_owned());
        }

        // sequential_flow をそのままループすれば大丈夫なはず
        for flow in self.sequential_flow.into_iter() {
            let index = flow.0;
            let destinations = &flow.1;
            let module = &mut self.modules[index];

            if module.necessary_parameter_value() != self.variables_stocks[index].len() {
                panic!(
                    "Mismatch parameter value:\n    necessary: {},\n    actual: {}.",
                    module.necessary_parameter_value(),
                    self.variables_stocks[index].len()
                );
            }

            loop_result = module.forward(
                &NNForwardInput {
                    inputs: self.variables_stocks[index]
                        .iter()
                        .map(|m| m.view())
                        .collect(),
                    target: Some(target.clone()),
                },
                is_grad,
            );

            for dst in destinations {
                self.variables_stocks[*dst].push(loop_result.clone());
            }
        }

        // 順伝播が終わったら、引数のリストは消去する必要がある
        for var in &mut self.variables_stocks {
            var.clear();
        }

        loop_result
    }

    fn propagate_grad(&mut self, grad: Option<&ndarray::ArrayViewD<T>>, eta: T) -> ArrayD<T> {
        let mut is_none = match grad {
            None => true,
            Some(_) => false,
        };
        let mut loop_result = ArrayD::<T>::zeros(vec![]);

        // inverse_sequential_flow をそのままループするだけで、目的は達成される
        for flow in self.inverse_sequential_flow.into_iter() {
            let index = flow.0;
            let destinations = &flow.1;
            let module = &mut self.modules[index];

            let loop_result_view = loop_result.view();
            loop_result = module.propagate_grad(
                if is_none {
                    None
                } else {
                    Some(&loop_result_view)
                },
                eta,
            );
            is_none = false;

            for dst in destinations {
                self.variables_stocks[*dst].push(loop_result.clone());
            }
        }

        // 逆伝播が終わったら、引数のリストは消去する必要がある
        for var in &mut self.variables_stocks {
            var.clear();
        }

        loop_result
    }

    impl_as_any_with_mut!();
}

impl<T> NNDataFlowTree<T>
where
    T: NNNecessaryTraits,
{
    pub fn new() -> Self {
        let modules = Vec::<Box<dyn NNModule<T>>>::new();
        let adjacency_list = Vec::<Vec<usize>>::new();
        let inverse_adjacency_list = Vec::<Vec<usize>>::new();
        let current_count = 0;
        let root_node_indices = Vec::<usize>::new();
        // let use_calculation_result_times = Vec::<usize>::new();
        let variables_stocks = Vec::<Vec<ArrayD<T>>>::new();
        // let input_variables_indices = Vec::<Vec<usize>>::new();
        let sequential_flow = NNSequentialNodeFlow::new();
        let inverse_sequential_flow = NNSequentialNodeFlow::new();
        let is_not_yet_executed = true;

        Self {
            modules,
            adjacency_list,
            inverse_adjacency_list,
            current_count,
            root_node_parameters: root_node_indices,
            // use_calculation_result_times,
            variables_stocks,
            // input_variables_indices,
            sequential_flow,
            inverse_sequential_flow,
            is_not_yet_executed,
        }
    }

    pub fn add_from_root<U>(&mut self, module: U) -> NNDataFlowNodeIndexInfo
    where
        T: Send + Sync + Debug + Float + ConstOne + ConstZero + 'static,
        U: NNModule<T> + 'static,
    {
        let parameter_value = module.necessary_parameter_value();
        for i in 0..parameter_value {
            self.modules.push(Box::new(InputTensor::<T> {
                phantom: PhantomData,
            }));
            self.adjacency_list.push(Vec::<usize>::new());
            self.adjacency_list[self.current_count + i].push(self.current_count + parameter_value);
        }
        self.current_count += 1;

        self.modules.push(Box::new(module));
        let index = self.current_count;
        self.current_count += 1;

        self.adjacency_list.push(Vec::<usize>::new());

        for i in 1..=parameter_value {
            self.root_node_parameters.push(index - i);
        }

        NNDataFlowNodeIndexInfo { index }
    }

    pub fn add<U>(
        &mut self,
        from: &NNDataFlowNodeIndexInfo,
        module: U,
    ) -> NNDataFlowNodeIndexInfo where U: NNModule<T> + 'static {
        self.modules.push(Box::new(module));
        let from_index = from.index;
        let to_index = self.current_count;
        self.current_count += 1;

        self.adjacency_list.push(Vec::<usize>::new());
        self.adjacency_list[from_index].push(to_index);

        NNDataFlowNodeIndexInfo { index: to_index }
    }

    pub fn calc_sequential_node_flow(&mut self)
    {
        let mut calculation_queue = VecDeque::<CalculationNodeState>::new();

        let queue_match =
            |source: usize, destination: usize, queue: &mut VecDeque<CalculationNodeState>| {
                match queue.iter().position(|state| state.id == destination) {
                    None => queue.push_back(CalculationNodeState {
                        id: destination,
                        args: vec![source],
                    }),
                    Some(pos) => queue.get_mut(pos).unwrap().args.push(source),
                }
            };

        // 入力パラメータがどのノードに渡されるのかを表現する
        for index in &self.root_node_parameters {
            for destination in &self.adjacency_list[*index] {
                self.sequential_flow.add(*index, vec![*destination]);
                // self.input_variables_indices[*destination].push(*index);

                queue_match(*index, *destination, &mut calculation_queue);
            }
        }

        let mut last_index: usize = 0;

        // キューが無くなるまで計算順序割り出し処理を継続する
        while calculation_queue.is_empty() == false {
            let current_state = calculation_queue.pop_front().unwrap();
            let current_id = current_state.id;
            let current_module = &self.modules[current_id];
            let necessary_parameter_value = current_module.necessary_parameter_value();
            let current_parameter_value = current_state.args.len();

            if current_parameter_value > necessary_parameter_value {
                panic!(
                    "Parameter value is Exceeded\n    necessary: {}\n    current: {}",
                    necessary_parameter_value, current_parameter_value
                );
            }
            if current_parameter_value < necessary_parameter_value {
                calculation_queue.push_back(current_state);
                continue;
            }

            // パラメータの数がちょうど求められていた数の場合は、self.sequential_flow に情報を入れ、calculation_queue に必要な情報を入れる
            let destinations = &self.adjacency_list[current_id];
            let push_index = self.sequential_flow.add(current_id, vec![]) - 1;

            for dst in destinations {
                queue_match(current_id, *dst, &mut calculation_queue);
                self.sequential_flow
                    .add_destination_to_index(push_index, *dst);
            }

            last_index = current_id;
        }

        // すべての辺の向きが逆方向であるグラフの隣接リストを作成する
        for dst in 0..self.adjacency_list.len() {
            self.inverse_adjacency_list
                .resize(self.modules.len(), Vec::<usize>::new());

            for from_id in &self.adjacency_list[dst] {
                self.inverse_adjacency_list[*from_id].push(dst);
            }
        }

        // 逆伝播時に最初に実行されるモジュールに関する処理
        for destination in &self.inverse_adjacency_list[last_index] {
            self.inverse_sequential_flow
                .add(last_index, vec![*destination]);
            // self.input_variables_indices[*destination].push(*index);

            queue_match(last_index, *destination, &mut calculation_queue);
        }

        // 同様に、キューが無くなるまで計算順序を割り出していけばいい
        while calculation_queue.is_empty() == false {
            let current_state = calculation_queue.pop_front().unwrap();
            let current_id = current_state.id;
            let current_module = &self.modules[current_id];
            let necessary_parameter_value = current_module.necessary_parameter_value();
            let current_parameter_value = current_state.args.len();

            if current_parameter_value > necessary_parameter_value {
                panic!(
                    "Parameter value is Exceeded\n    necessary: {}\n    current: {}",
                    necessary_parameter_value, current_parameter_value
                );
            }
            if current_parameter_value < necessary_parameter_value {
                calculation_queue.push_back(current_state);
                continue;
            }

            // パラメータの数がちょうど求められていた数の場合は、self.inverse_sequential_flow に情報を入れ、calculation_queue に必要な情報を入れる
            let destinations = &self.inverse_adjacency_list[current_id];
            let push_index = self.inverse_sequential_flow.add(current_id, vec![]) - 1;

            for dst in destinations {
                queue_match(current_id, *dst, &mut calculation_queue);
                self.inverse_sequential_flow
                    .add_destination_to_index(push_index, *dst);
            }
        }
    }

    pub fn access_module(&self, info: &NNDataFlowNodeIndexInfo) -> &dyn NNModule<T> {
        self.modules[info.index].as_ref()
    }

    pub fn access_module_mut(&mut self, info: &NNDataFlowNodeIndexInfo) -> &mut dyn NNModule<T> {
        self.modules[info.index].as_mut()
    }
}
