from typing import Dict, Any, Optional, Tuple, List
import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import networkx as nx
import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.data import Composite, Bounded, Unbounded, TensorSpec
import traceback
from graph_gym import _collect_attribute_vocabulary_and_max_len, _try_convert_string_to_number
# NOTE Compatiability Issues: TorchRL 0.5 requires PyTorch 2.4.0, and TorchRL 0.6 requires PyTorch 2.5.0.

def _encode_and_pad_attributes(
    attributes: Dict[str, Any],
    all_keys: List[str],
    vocab_mapping: Dict[str, int],
    effective_max_len: int,
    padding_value: float
) -> np.ndarray:
    """Encodes and pads attributes into a fixed-size numerical vector."""
    encoded_padded_attributes = []
    for key in all_keys:
        value = attributes.get(key, None)
        processed_value = []
        if value is None: processed_value = [padding_value] * effective_max_len
        else:
            items_to_process = value if isinstance(value, list) else [value]
            for item in items_to_process:
                evaluated_item = _try_convert_string_to_number(item)
                if isinstance(evaluated_item, str):
                    encoded_value = vocab_mapping.get(evaluated_item, padding_value)
                    processed_value.append(float(encoded_value))
                elif isinstance(evaluated_item, (int, float, np.number)): processed_value.append(float(evaluated_item))
                else: processed_value.append(padding_value)
            while len(processed_value) < effective_max_len: processed_value.append(padding_value)
            processed_value = processed_value[:effective_max_len]
        encoded_padded_attributes.extend(processed_value)
    expected_flat_len = len(all_keys) * effective_max_len
    if len(encoded_padded_attributes) != expected_flat_len:
        print(f"Warning: Generated attribute vector length ({len(encoded_padded_attributes)}) does not match expected length ({expected_flat_len}).")
        while len(encoded_padded_attributes) < expected_flat_len: 
            encoded_padded_attributes.append(padding_value)
        encoded_padded_attributes = encoded_padded_attributes[:expected_flat_len]
    return np.array(encoded_padded_attributes, dtype=np.float32)


class SubstratumGraphEnv(gym.Env):
    """
    SubstratumGraphEnv
    
    A custom OpenAI Gymnasium environment that models a directed graph (DiGraph) as its state space. 
    The environment allows an agent to navigate through the graph by selecting target nodes as actions. 
    The transitions are validated based on graph adjacency, and rewards are determined by edge weights.
    Attributes:
        _initial_nx_graph (nx.DiGraph): The initial NetworkX directed graph representing the environment's state.
        _current_nx_graph (nx.DiGraph): A copy of the initial graph representing the current state of the environment.
        observation_space (spaces.Graph): The observation space, conforming to gymnasium.spaces.Graph.
        action_space (spaces.Discrete): The action space, representing the indices of nodes in the graph.
        _num_nodes_in_graph (int): The number of nodes in the graph.
        _enable_debugging_prints (bool): Flag to enable or disable debugging prints.
        _vocab_mapping (dict): A mapping of attribute values to encoded indices for vocabulary-based encoding.
        _max_list_len (int): The maximum length of list attributes in the graph.
        _effective_max_len (int): The effective maximum length for padding attributes.
        _all_node_attr_keys (list): A sorted list of all node attribute keys in the graph.
        _all_edge_attr_keys (list): A sorted list of all edge attribute keys in the graph.
        _padding_value (float): The padding value used for encoding attributes.
        _current_node_idx (Optional[int]): The current node index where the agent is positioned.
        _node_to_index (dict): A mapping from node names to their corresponding indices.
        _index_to_node (dict): A mapping from indices to their corresponding node names.
        _explicit_start_node_index (Optional[int]): The explicitly defined start node index, if provided.
        _terminal_node_index (Optional[int]): The explicitly defined terminal node index, if provided.
        _episode_finished (bool): A flag indicating whether the episode has finished.
    Methods:
        __init__(initial_nx_graph, observation_space, start_node_name=None, terminal_node_name=None, enable_debugging_prints=True):
            Initializes the environment with a NetworkX graph, observation space, and optional start/terminal nodes.
        _get_observation_from_nx_graph():
            Converts the current NetworkX graph state into a gymnasium.spaces.Graph observation dictionary format.
        reset(seed=None, options=None):
            Resets the environment to its initial state and returns the initial observation and info.
        step(action):
            Executes a step in the environment by attempting to move to the target node specified by the action.
            Returns the next observation, reward, termination flag, truncation flag, and additional info.
        render():
            Renders the environment (not implemented).
        close():
            Cleans up resources used by the environment (not implemented).
    """
    def __init__(self, initial_nx_graph: nx.DiGraph, 
                 observation_space: spaces.Graph, 
                 start_node_name: Optional[Any] = None,
                 terminal_node_name: Optional[Any] = None, 
                 enable_debugging_prints: bool = True):
        """Initializes the environment with a NetworkX graph, observation space, and optional start/terminal nodes."""
        super().__init__()
        if not isinstance(initial_nx_graph, nx.DiGraph): 
            raise TypeError("initial_nx_graph must be a NetworkX DiGraph.")
        if not isinstance(observation_space, spaces.Graph): 
            raise TypeError("observation_space must be a gymnasium.spaces.Graph.")
        if not isinstance(observation_space.node_space, spaces.Box): 
            raise TypeError("Node space within observation_space must be a Box space.")
        if observation_space.edge_space is not None and not isinstance(observation_space.edge_space, spaces.Box): 
            raise TypeError("Edge space within observation_space must be a Box space or None.")
        self._initial_nx_graph = initial_nx_graph
        self._current_nx_graph = initial_nx_graph.copy() # Environment state is a copy
        self.observation_space = observation_space
        self._num_nodes_in_graph = self._current_nx_graph.number_of_nodes()
        self._enable_debugging_prints = enable_debugging_prints

        if self._num_nodes_in_graph > 0:
            self.action_space = spaces.Discrete(self._num_nodes_in_graph)
        else:
            self.action_space = spaces.Discrete(1)
        self._vocab_mapping, self._max_list_len = _collect_attribute_vocabulary_and_max_len(self._initial_nx_graph)
        self._effective_max_len = max(1, self._max_list_len)
        self._all_node_attr_keys = sorted(list(set().union(*(d.keys() for _, d in self._initial_nx_graph.nodes(data=True)))))
        self._all_edge_attr_keys = sorted(list(set().union(*(d.keys() for _, _, d in self._initial_nx_graph.edges(data=True)))))
        self._padding_value = -999.0 # Consistent padding value
        self._current_node_idx: Optional[int] = None
        sorted_nodes = sorted(self._current_nx_graph.nodes())
        self._node_to_index = {node: i for i, node in enumerate(sorted_nodes)}
        self._index_to_node = {i: node for node, i in self._node_to_index.items()}
        self._explicit_start_node_index: Optional[int] = None
        if start_node_name is not None:
            if start_node_name in self._node_to_index:
                self._explicit_start_node_index = self._node_to_index[start_node_name]
            else: print(f"Warning: Explicit start node name '{start_node_name}' not found in the graph. Will sample start node randomly.")
        self._terminal_node_index: Optional[int] = None
        if terminal_node_name is not None:
            if terminal_node_name in self._node_to_index: 
                self._terminal_node_index = self._node_to_index[terminal_node_name]
            else:
                print(f"Warning: Terminal node name '{terminal_node_name}' not found in the graph. No terminal node set.")
        self._episode_finished = False


    def _get_observation_from_nx_graph(self) -> Dict[str, np.ndarray]:
        """Converts the current NetworkX graph state into a gymnasium.spaces.Graph observation dictionary format."""
        nx_graph = self._current_nx_graph
        num_nodes = nx_graph.number_of_nodes()
        num_edges = nx_graph.number_of_edges()
        if not nx_graph:
            obs = {
                 "nodes": np.zeros((0, self.observation_space.nodes.shape[0]), dtype=np.float32),
                 "edge_links": np.zeros((0, 2), dtype=np.int64),
                 "num_nodes": np.array(0, dtype=np.int64),
                 "num_edges": np.array(0, dtype=np.int64)}
            if self.observation_space.edge_space is not None: 
                obs["edges"] = np.zeros((0, self.observation_space.edge_space.shape[0]), dtype=np.float32)
            return obs
        node_features_list = []
        node_to_index = self._node_to_index
        sorted_nodes = sorted(nx_graph.nodes(), key=lambda node: node_to_index[node])
        for node in sorted_nodes:
            attrs = nx_graph.nodes[node]
            encoded_features = _encode_and_pad_attributes(
                attrs,
                self._all_node_attr_keys,
                self._vocab_mapping,
                self._effective_max_len,
                self._padding_value
            )
            node_features_list.append(encoded_features)
        nodes_data = np.stack(node_features_list)
        edge_features_list = []
        edge_links_data = np.zeros((num_edges, 2), dtype=np.int64)
        for i, (u, v, attrs) in enumerate(nx_graph.edges(data=True)):
            edge_links_data[i, 0] = node_to_index[u]
            edge_links_data[i, 1] = node_to_index[v]
            if self.observation_space.edge_space is not None:
                encoded_features = _encode_and_pad_attributes(
                    attrs,
                    self._all_edge_attr_keys,
                    self._vocab_mapping,
                    self._effective_max_len,
                    self._padding_value
                )
                edge_features_list.append(encoded_features)
        edges_data = None
        if self.observation_space.edge_space is not None:
            if edge_features_list: 
                edges_data = np.stack(edge_features_list)
            else:
                 edge_feature_dim = self.observation_space.edge_space.shape[0]
                 edges_data = np.zeros((0, edge_feature_dim), dtype=np.float32)
        observation = {
            "nodes": nodes_data,
            "edge_links": edge_links_data,
            "num_nodes": np.array(num_nodes, dtype=np.int64),
            "num_edges": np.array(num_edges, dtype=np.int64)
        }
        if edges_data is not None:
            observation["edges"] = edges_data
        return observation


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Resets the environment to its initial state."""
        super().reset(seed=seed)
        self._current_nx_graph = self._initial_nx_graph.copy() 
        self._episode_finished = False 
        if self._num_nodes_in_graph > 0: 
            if self._explicit_start_node_index is not None: self._current_node_idx = self._explicit_start_node_index
            else:
                available_start_indices = list(range(self._num_nodes_in_graph))
                if self._terminal_node_index is not None and self._terminal_node_index in available_start_indices: available_start_indices.remove(self._terminal_node_index)
                if available_start_indices: self._current_node_idx = self.np_random.choice(available_start_indices)
                elif self._terminal_node_index is not None: self._current_node_idx = self._terminal_node_index 
                else: self._current_node_idx = self.action_space.sample(self.np_random)
        else:
            self._current_node_idx = None 
        self._current_graph_data = self._get_observation_from_nx_graph() 
        info = {"current_node_index": self._current_node_idx} 
        return self._current_graph_data, info


    def step(self, action: Any) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        if self._episode_finished:
            return self._current_graph_data, 0.0, True, False, {"current_node_index": self._current_node_idx, "step_after_termination": True}
        target_node_idx = int(action) 
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        is_valid_move = False

        current_nx_node_name = self._index_to_node.get(self._current_node_idx, "Unknown Node")
        target_nx_node_name = self._index_to_node.get(target_node_idx, "Invalid Index")
        if self._enable_debugging_prints:
            print(f"Debug: SubstratumGraphEnv step: Agent at '{current_nx_node_name}' (index {self._current_node_idx}), attempting to move to '{target_nx_node_name}' (index {target_node_idx}).")

        if self._current_node_idx is None or self._num_nodes_in_graph == 0:
            reward = -1.0
            is_valid_move = False
            info["error_message"] = "Agent not positioned or graph is empty."
            if self._enable_debugging_prints:
                print(f"Debug: SubstratumGraphEnv step: Invalid move (no agent position or empty graph). Reward: {reward}")
        elif target_node_idx < 0 or target_node_idx >= self._num_nodes_in_graph:
            reward = -1.0
            is_valid_move = False
            info["error_message"] = f"Invalid target node index: {target_node_idx}"
            if self._enable_debugging_prints:
                print(f"Debug: SubstratumGraphEnv step: Invalid target node index {target_node_idx}. Reward: {reward}")
        else:
            current_nx_node = self._index_to_node[self._current_node_idx]
            target_nx_node = self._index_to_node[target_node_idx]
            
            if self._current_nx_graph.has_edge(current_nx_node, target_nx_node):
                is_valid_move = True
                self._current_node_idx = target_node_idx
                edge_attributes = self._current_nx_graph.get_edge_data(current_nx_node, target_nx_node)
                edge_weight = edge_attributes.get('weight', 1.0) 
                if self._enable_debugging_prints:
                     print(f"Debug: SubstratumGraphEnv step: Valid move from {current_nx_node} to {target_nx_node}. Raw edge_weight: {edge_weight}")
                try:
                    reward = float(edge_weight)
                    if not np.isfinite(reward): 
                        print(f"Warning: Edge ({current_nx_node}, {target_nx_node}) has non-finite weight '{edge_weight}' after conversion to float. Defaulting reward to 0.")
                        reward = 0.0 
                except (ValueError, TypeError):
                    print(f"Warning: Edge ({current_nx_node}, {target_nx_node}) has non-numerical weight '{edge_weight}'. Defaulting reward to 100.")
                    reward = 100.0
                info["transitioned_to_node_index"] = self._current_node_idx
                if self._terminal_node_index is not None and self._current_node_idx == self._terminal_node_index:
                    terminated = True
                    self._episode_finished = True 
                    reward += 5000.0 
                    if self._enable_debugging_prints:
                         print(f"Debug: SubstratumGraphEnv step: Reached terminal node {target_nx_node}. Adding bonus {5000.0}. Total reward: {reward}")
            else:
                reward = -0.5
                is_valid_move = False
                info["transition_failed_target_index"] = target_node_idx
                info["error_message"] = f"No edge from node {current_nx_node} (index {self._current_node_idx}) to node {target_nx_node} (index {target_node_idx})."
                if self._enable_debugging_prints:
                    print(f"Debug: SubstratumGraphEnv step: Invalid move (no edge) from {current_nx_node} to {target_nx_node}. Reward: {reward}")
        info["is_valid_move"] = is_valid_move
        self._current_graph_data = self._get_observation_from_nx_graph()
        info["current_node_index"] = self._current_node_idx

        if self._enable_debugging_prints:
            print(f"Debug: SubstratumGraphEnv step: Step finished. Returning reward: {reward}, terminated: {terminated}, truncated: {truncated}")
        return self._current_graph_data, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass


class SubstratumBridge(EnvBase):
    """
    A manual wrapper for a gymnasium environment (specifically SubstratumGraphEnv)
    to make it compatible with TorchRL's data structures and API.

    This wrapper is designed to handle compatibility issues between gymnasium and TorchRL,
    particularly for environments with complex observation spaces like graphs. It provides
    methods for converting gymnasium spaces into TorchRL specs, resetting the environment,
    stepping through it, and managing device compatibility.

    Key Features:
    - Converts gymnasium spaces (e.g., `spaces.Box`, `spaces.Discrete`, `spaces.Graph`) into
      TorchRL specs (`Bounded`, `Unbounded`, `Composite`).
    - Handles dynamic dimensions in spaces (e.g., `None` in gymnasium shapes) by converting
      them to TorchRL-compatible formats.
    - Supports graph-based observation spaces with nodes, edges, edge links, and counts.
    - Converts gymnasium-style outputs (NumPy arrays, scalars) into TorchRL TensorDicts.
    - Manages device compatibility (CPU/GPU) for all tensors.

    Attributes:
    - `observation_spec`: TorchRL specification for the observation space.
    - `action_spec`: TorchRL specification for the action space.
    - `reward_spec`: TorchRL specification for the reward space.
    - `_gym_observation_space`: Original gymnasium observation space for reference.
    - `_gym_action_space`: Original gymnasium action space for reference.
    - `_device`: The device (CPU/GPU) used for tensors.

    Methods:
    - `__init__`: Initializes the wrapper with the underlying `SubstratumGraphEnv` and
      converts its spaces to TorchRL specs.
    - `_convert_gym_space_to_torchrl_spec`: Converts a gymnasium space into a TorchRL spec.
    - `_to_tensordict`: Converts gymnasium outputs (observations, rewards, etc.) into a
      TorchRL TensorDict.
    - `_reset`: Resets the environment and returns the initial state as a TensorDict.
    - `_step`: Executes a step in the environment using an action from a TensorDict.
    - `_set_seed`: Sets the random seed for the environment.
    - `device`: Property to get the device used for tensors.
    - `NotImplementedError`: If a gymnasium space type is not supported for conversion.
    This wrapper is particularly useful for reinforcement learning tasks involving
    graph-based environments, where observations and actions have complex structures
    that need to be seamlessly integrated with TorchRL's framework.
    """
    def __init__(self, initial_nx_graph: nx.DiGraph,
                observation_space: spaces.Graph,
                start_node_name: Optional[Any] = None,
                terminal_node_name: Optional[Any] = None,
                device: Optional[torch.device] = None,
                enable_debug_print: bool = False,
                **kwargs):
        """Initializes the  SubstratumBridge."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        super().__init__(device=self.device, batch_size=[]) 
        self._env = SubstratumGraphEnv(initial_nx_graph=initial_nx_graph, observation_space=observation_space, start_node_name=start_node_name, terminal_node_name=terminal_node_name)
        inner_obs_spec = self._convert_gym_space_to_torchrl_spec(self._env.observation_space)
        observation_spec_dict = {
            "observation": inner_obs_spec,
        }
        self.observation_spec = Composite(observation_spec_dict, device=self.device)
        self._enable_debug_print = enable_debug_print
        action_spec_inner = self._convert_gym_space_to_torchrl_spec(self._env.action_space)
        composite_action_spec = Composite({"action": action_spec_inner}, device=self.device)
        self.full_action_spec = composite_action_spec
        if  self._enable_debug_print:
            print(f"Debug:  SubstratumBridge __init__ finished: type(self.action_spec) = {type(self.action_spec)}")
            if isinstance(self.action_spec, TensorSpec):
                print(f"Debug:  SubstratumBridge __init__ finished: self.action_spec shape = {self.action_spec.shape}")
                print(f"Debug:  SubstratumBridge __init__ finished: self.action_spec dtype = {self.action_spec.dtype}")
            print(f"Debug:  SubstratumBridge __init__ finished: type(self.full_action_spec) = {type(self.full_action_spec)}")
            if isinstance(self.full_action_spec, Composite):
                print(f"Debug:  SubstratumBridge __init__ finished: self.full_action_spec keys = {list(self.full_action_spec.keys())}")
                if "action" in self.full_action_spec.keys():
                    full_action_spec_inner_final = self.full_action_spec["action"]
                    print(f"Debug:  SubstratumBridge __init__ finished: self.full_action_spec['action'] type = {type(full_action_spec_inner_final)}")
                    if isinstance(full_action_spec_inner_final, TensorSpec):
                        print(f"Debug:  SubstratumBridge __init__ finished: self.full_action_spec['action'] shape = {full_action_spec_inner_final.shape}")
                        print(f"Debug:  SubstratumBridge __init__ finished: self.full_action_spec['action'] dtype = {full_action_spec_inner_final.dtype}")
        self._gym_observation_space = self._env.observation_space
        self._gym_action_space = self._env.action_space


    def _convert_gym_space_to_torchrl_spec(self, gym_space: gym.Space) -> TensorSpec:
        """Converts a gymnasium space object into a corresponding TorchRL TensorSpec or CompositeSpec."""
        if isinstance(gym_space, spaces.Box):
            torchrl_shape = tuple(d if d is not None else -1 for d in gym_space.shape)
            low = torch.tensor(gym_space.low, dtype=torch.float32)
            high = torch.tensor(gym_space.high, dtype=torch.float32)
            if torch.isinf(low).any() or torch.isinf(high).any():
                return Unbounded(shape=torchrl_shape, dtype=torch.float32, device=self.device)
            else:
                return Bounded(low=low, high=high, shape=torchrl_shape, dtype=torch.float32, device=self.device)
        elif isinstance(gym_space, spaces.Discrete):
            low = torch.tensor(0, dtype=torch.int64)
            high = torch.tensor(gym_space.n - 1, dtype=torch.int64)
            return Bounded(low=low, high=high, shape=(1,), dtype=torch.int64, device=self.device)
        elif isinstance(gym_space, spaces.Graph):
            obs_dict_spec = {}
            if gym_space.node_space is not None:
                obs_dict_spec["nodes"] = self._convert_gym_space_to_torchrl_spec(gym_space.node_space)
            else:
                 print("Warning: Gymnasium Graph observation_space has node_space=None. Defining placeholder spec.")
                 obs_dict_spec["nodes"] = Unbounded(shape=(-1, 1), dtype=torch.float32, device=self.device) 
            if gym_space.edge_space is not None:
                obs_dict_spec["edges"] = self._convert_gym_space_to_torchrl_spec(gym_space.edge_space)
            num_nodes_in_graph = getattr(self._env, '_num_nodes_in_graph', 0)
            low_links = torch.tensor(0, dtype=torch.int64)
            high_links = torch.tensor(num_nodes_in_graph - 1 if num_nodes_in_graph > 0 else 0, dtype=torch.int64)
            obs_dict_spec["edge_links"] = Bounded(low=low_links, high=high_links, shape=(-1, 2), dtype=torch.int64, device=self.device)
            low_scalar = torch.tensor(0, dtype=torch.int64)
            high_num_nodes = torch.tensor(num_nodes_in_graph, dtype=torch.int64)
            obs_dict_spec["num_nodes"] = Bounded(low=low_scalar, high=high_num_nodes, shape=(1,), dtype=torch.int64, device=self.device)
            obs_dict_spec["num_edges"] = Unbounded(shape=(1,), dtype=torch.int64, device=self.device) 
            return Composite(obs_dict_spec, device=self.device)
        else:
            raise NotImplementedError(f"Conversion for gymnasium space type {type(gym_space)} not implemented.")


    def _to_tensordict(self, observation: Dict[str, np.ndarray], reward: float, terminated: bool, truncated: bool, info: Dict[str, Any]) -> TensorDict:
        """Converts gymnasium outputs (NumPy arrays, scalars) to a TensorDict."""
        obs_td_data = {}
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                if value.dtype == object:
                     print(f"Warning: Observation key '{key}' has object dtype. Attempting conversion to string.")
                     value = np.array([str(x) for x in value.flatten()]).reshape(value.shape)
                if value.dtype in [np.float32, np.float64, np.int32, np.int64, np.bool_]:
                     obs_td_data[key] = torch.from_numpy(value).to(self.device)
                else:
                     print(f"Warning: Observation key '{key}' has unsupported dtype {value.dtype}. Skipping conversion to tensor.")
                     obs_td_data[key] = value 
            elif isinstance(value, (int, float, bool, np.number)):
                obs_td_data[key] = torch.tensor(value, device=self.device)
            else:
                obs_td_data[key] = value
        obs_td = TensorDict(obs_td_data, batch_size=[])
        step_td = TensorDict({
            "observation": obs_td,
            "reward": torch.tensor([reward], dtype=torch.float32, device=self.device),
            "terminated": torch.tensor([terminated], dtype=torch.bool, device=self.device),
            "truncated": torch.tensor([truncated], dtype=torch.bool, device=self.device),
            "info": info 
        }, batch_size=[])
        return step_td


    def _reset(self, tensordict: TensorDict, **kwargs) -> TensorDict:
        """Resets the environment to an initial state."""
        observation, info = self._env.reset(**kwargs)
        reset_td = self._to_tensordict(observation, 0.0, False, False, info)
        return reset_td


    def _step(self, tensordict: TensorDict) -> TensorDict:
        """Takes an action (from a TensorDict) and steps the environment."""
        action = tensordict["action"].squeeze().long().cpu().numpy().item()
        observation, reward, terminated, truncated, info = self._env.step(action)
        step_td = self._to_tensordict(observation, reward, terminated, truncated, info)
        return step_td

    def _set_seed(self, seed: Optional[int]):
        """Sets the seed for the environment."""
        self._env.reset(seed=seed)

    @property
    def device(self) -> torch.device:
        """Returns the device used by the environment."""
        return self._device
