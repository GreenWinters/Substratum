# -*- coding: utf-8 -*-
import os
import csv
import datetime
import time
import traceback
import torch
import sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx
from tensordict import TensorDict
from torchrl.data import Composite, Bounded, TensorSpec
from typing import Tuple, Optional, Dict, Any
from tqdm import tqdm
from net_analysis import graph_construction
from torch_rl_env import  SubstratumBridge
from graph_gym import convert_nx_digraph_to_gym_graph


try:
    from torch_geometric.nn import GCNConv, global_mean_pool
    _torch_geometric_available = True
except ImportError:
    print("Warning: torch_geometric not found. Graph Neural Network layers will not be available.")
    _torch_geometric_available = False


#Graph A2C Model Definition
class GraphA2C(nn.Module):
    """
    An A2C model that processes graph observations using GNNs and outputs
    action probabilities (actor) and a value estimate (critic).
    """
    def __init__(self, observation_spec: Composite, action_spec: Bounded, output_dim: int = 64):
        """
        This model is designed to process graph-structured data using Graph Convolutional Networks (GCN) 
        and implements an Actor-Critic architecture for reinforcement learning.

            observation_spec (Composite): The TorchRL Composite observation spec containing the inner graph data. 
                Must include a key 'observation' with a nested spec containing a key 'nodes' for node features.
            action_spec (Bounded): The TorchRL Bounded action spec defining the discrete action space. 
                Must have dtype `torch.int64` and a scalar shape (() or (1,)).
            output_dim (int, optional): The dimension of the shared graph representation vector. Defaults to 64.

        Raises:
            RuntimeError: If `torch_geometric` is not available.
            ValueError: If `observation_spec` does not contain a key 'observation' or if the inner graph spec 
                does not contain a key 'nodes'.
            TypeError: If `action_spec` is not a Bounded spec or if its dtype is not `torch.int64`.
            ValueError: If `action_spec` does not have a scalar shape (() or (1,)).
        """
        super().__init__()
        if not _torch_geometric_available:
            raise RuntimeError("torch_geometric is not available. Cannot initialize GraphA2C.")
        if "observation" not in observation_spec:
            raise ValueError("observation_spec must contain a key 'observation' for the inner graph spec.")
        inner_graph_obs_spec = observation_spec["observation"]
        # Get the shape of node features from the inner graph observation spec
        if "nodes" not in inner_graph_obs_spec:
            raise ValueError("Inner graph observation spec must contain a key 'nodes'.")
        node_feature_dim = inner_graph_obs_spec["nodes"].shape[-1]
        # Get the number of discrete actions from the Bounded action spec
        if not isinstance(action_spec, Bounded):
            raise TypeError("action_spec must be a Bounded spec for discrete actions.")
        if action_spec.dtype != torch.int64:
            raise TypeError("action_spec dtype must be torch.int64 for discrete actions.")
        # Check if the action space is scalar (shape is ()) or (1,)
        if action_spec.shape not in [torch.Size([]), torch.Size([1])]:
            raise ValueError(f"action_spec shape must be () or (1,) for a single discrete action, but got {action_spec.shape}.")
        # Calculate the number of actions from the bounds
        num_actions = int(action_spec.high.item() - action_spec.low.item() + 1)
        # Shared Graph Feature Extractor  - Uses GCNConv layers to process the graph structure and node features. Edge adjacency is not used in this basic GCNConv implementation.
        self.node_fc = nn.Linear(node_feature_dim, output_dim)
        self.conv1 = GCNConv(output_dim, output_dim)
        self.conv2 = GCNConv(output_dim, output_dim)
        # NOTE For future studies - Add more GNN layers as needed
        # --- Actor (Policy) takes the graph representation and outputs logits for each action, selecting each node.
        self.actor = nn.Linear(output_dim, num_actions)
        # Critic (Value) takes the graph representation and outputs a single value estimate.
        self.critic = nn.Linear(output_dim, 1)

    def forward(self, observation: TensorDict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes a graph observation TensorDict to output action logits and value estimate.
        Args:
            observation: A TensorDict containing the graph observation.
                         Expected keys: "observation" (nested TensorDict) with
                         "nodes", "edge_links", "num_nodes", "num_edges",
                         and optionally "edges".
        Returns:
            A tuple containing:
            - Action logits (for policy).
            - Value estimate (for critic).
        """
        if not _torch_geometric_available:
            raise RuntimeError("torch_geometric is not available. Cannot run GraphA2C forward pass.")
        # Extract data from the observation TensorDict - Access nested observation data correctly from the input TensorDict
        node_features = observation["observation"]["nodes"]
        # Ensure edge_links is in the correct format (shape [2, num_edges])
        edge_links = observation["observation"]["edge_links"].t().contiguous()
        # Process node features
        x = F.relu(self.node_fc(node_features))
        # Apply GNN layers
        # edge_index is the standard name for edge_links in torch_geometric
        edge_index = edge_links
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training) # Optional dropout
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        # Add more layers here
        # Global pooling to get a fixed-size graph representation
        batch_index = torch.zeros(node_features.size(0), dtype=torch.long, device=node_features.device)
        graph_representation = global_mean_pool(x, batch_index)
        # Get action logits from the actor head
        action_logits = self.actor(graph_representation)
        # Get value estimate from the critic head
        value_estimate = self.critic(graph_representation)
        return action_logits, value_estimate


class GraphA2CTrainer:
    """
    Trains a GraphA2C model using the A2C algorithm with a graph environment.
    Includes debug print statements to help diagnose tensor and TensorDict issues.
    """
    def __init__(
        self,
        model: nn.Module, # Use nn.Module for type hint as GraphA2C is defined elsewhere
        env:  SubstratumBridge,
        optimizer: optim.Optimizer,
        gamma: float = 0.99, # Discount factor
        value_loss_coef: float = 0.5, # Coefficient for the value loss
        entropy_coef: float = 0.01, # Coefficient for the entropy term
        n_steps: int = 1, # Number of steps to collect before performing an update (standard A2C is n=1)
        device: Optional[torch.device] = None,
        enable_debugging_prints: bool = False
    ):
        """
        Initializes the GraphA2CTrainer.

        Args:
            model: The GraphA2C model to train.
            env: The  SubstratumBridge environment.
            optimizer: The PyTorch optimizer for the model.
            gamma: Discount factor for future rewards.
            value_loss_coef: Weight for the value function loss.
            entropy_coef: Weight for the policy entropy term.
            n_steps: Number of steps to collect before an update.
            device: The PyTorch device to use.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = model.to(self.device) # Ensure model is on the correct device
        self.env = env # Environment should already be on the correct device
        self.optimizer = optimizer
        self.gamma = gamma
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.n_steps = n_steps
        self._enable_debugging_prints =  enable_debugging_prints
        if self.env.device != self.device:
            print(f"Warning: Environment device ({self.env.device}) does not match Trainer device ({self.device}). This may cause issues.")
        
        if self._enable_debugging_prints:
            print(f"Debug: GraphA2CTrainer __init__: type(self.env.action_spec) = {type(self.env.action_spec)}")
            if isinstance(self.env.action_spec, TensorSpec):
                print(f"Debug: GraphA2CTrainer __init__: self.env.action_spec shape = {self.env.action_spec.shape}")
                print(f"Debug: GraphA2CTrainer __init__: self.env.action_spec dtype = {self.env.action_spec.dtype}")
            if isinstance(self.env.full_action_spec, Composite):
                print(f"Debug: GraphA2CTrainer __init__: self.env.action_spec keys = {list(self.env.full_action_spec.keys())}")
                if "action" in self.env.full_action_spec.keys():
                    action_spec_inner = self.env.full_action_spec["action"]
                    print(f"Debug: GraphA2CTrainer __init__: self.env.full_action_spec['action'] type = {type(action_spec_inner)}")
                    if isinstance(action_spec_inner, TensorSpec):
                        print(f"Debug: GraphA2CTrainer __init__: self.env.action_spec['action'] shape = {action_spec_inner.shape}")
                        print(f"Debug: GraphA2CTrainer __init__: self.env.action_spec['action'] dtype = {action_spec_inner.dtype}")
                else:
                    print("Debug: GraphA2CTrainer __init__: 'action' key not found in self.env.action_spec Composite.")
            else:
                if isinstance(self.env.action_spec, TensorSpec):
                    print(f"Debug: GraphA2CTrainer __init__: self.env.action_spec shape = {self.env.action_spec.shape}")
                    print(f"Debug: GraphA2CTrainer __init__: self.env.action_spec dtype = {self.env.action_spec.dtype}")
                else:
                    print(f"Debug: GraphA2CTrainer __init__: self.env.action_spec is not a TensorSpec.")
            print(f"Debug: GraphA2CTrainer __init__: type(self.env.full_action_spec) = {type(self.env.full_action_spec)}")
            if isinstance(self.env.full_action_spec, Composite): 
                print(f"Debug: GraphA2CTrainer __init__: self.env.full_action_spec keys = {list(self.env.full_action_spec.keys())}")
                if "action" in self.env.full_action_spec.keys():
                    full_action_spec_inner = self.env.full_action_spec["action"]
                    print(f"Debug: GraphA2CTrainer __init__: self.env.full_action_spec['action'] type = {type(full_action_spec_inner)}")
                    if isinstance(full_action_spec_inner, TensorSpec):
                        print(f"Debug: GraphA2CTrainer __init__: self.env.full_action_spec['action'] shape = {full_action_spec_inner.shape}")
                        print(f"Debug: GraphA2CTrainer __init__: self.env.full_action_spec['action'] dtype = {full_action_spec_inner.dtype}")
                else:
                    print("Debug: GraphA2CTrainer __init__: 'action' key not found in self.env.full_action_spec Composite.")
            else:
                print("Debug: GraphA2CTrainer __init__: self.env.full_action_spec is not a Composite.")
        if not isinstance(self.env.full_action_spec, Composite) or "action" not in self.env.full_action_spec.keys():
            raise TypeError(
                f"Expected env.full_action_spec to be a Composite spec with an 'action' key, "
                f"but got type {type(self.env.full_action_spec)} with keys "
                f"{list(self.env.full_action_spec.keys()) if isinstance(self.env.full_action_spec, Composite) else 'N/A'}. "
                f"Ensure  SubstratumBridge correctly sets full_action_spec.")

    def calculate_returns(self, rewards: torch.Tensor, terminated: torch.Tensor, last_value: torch.Tensor) -> torch.Tensor:
        """
        Calculates the discounted returns for a sequence of rewards.
        Args:
            rewards: Tensor of rewards for steps in a trajectory (shape [T, 1]).
            terminated: Boolean tensor indicating if the episode terminated at each step (shape [T, 1]).
            last_value: The value estimate of the state *after* the last step (shape [1, 1]).
        Returns:
            Tensor of discounted returns (shape [T, 1]).
        """
        returns = torch.zeros_like(rewards)
        for t in reversed(range(rewards.size(0))):
            if terminated[t].item():
                running_return = rewards[t].squeeze(-1) 
            else:
                running_return = rewards[t].squeeze(-1) + self.gamma * running_return
            returns[t] = running_return 
        return returns


    def train_step(self, current_td: TensorDict) -> Tuple[Optional[TensorDict], Dict[str, float]]:
        """
        Performs one training step: collects n_steps of data, calculates losses, and updates the model.
        Includes extensive debug prints to help diagnose issues with tensors and TensorDicts.

        Args:
            current_td: The initial TensorDict for this training segment (from env.reset() or previous env.step()).

        Returns:
            A tuple containing:
            - The TensorDict representing the state after the last step in this segment,
              or None if an error occurred during the step collection.
            - A dictionary containing training metrics (e.g., total_loss, policy_loss, value_loss, entropy).
        """
        log_probs = []
        values = []
        rewards = []
        terminated_flags = []
        truncated_flags = []
        infos = []
        trajectory = []
        entropies = [] 
        for step in range(self.n_steps):
            try:
                if self._enable_debugging_prints:
                    print(f"Debug: Step {step}, Start of loop: current_td batch_size = {current_td.batch_size}")
                action_logits, value_estimate = self.model(current_td)
                if self._enable_debugging_prints:
                    print(f"Debug: Step {step}, Model Output: action_logits shape = {action_logits.shape}, value_estimate shape = {value_estimate.shape}")
                    print(f"Debug: Step {step}, Model Output: action_logits (first 10 elements) = {action_logits[0, :10] if action_logits.numel() > 0 else 'Empty Logits'}")
                action_dist = torch.distributions.Categorical(logits=action_logits)
                action_tensor = action_dist.sample()

                if self._enable_debugging_prints:
                    print(f"Debug: Step {step}, Before log_prob: action_dist batch_shape = {action_dist.batch_shape}")
                    print(f"Debug: Step {step}, Before log_prob: action_dist event_shape = {action_dist.event_shape}")
                    print(f"Debug: Step {step}, Before log_prob: action_tensor shape = {action_tensor.shape}")
                    print(f"Debug: Step {step}, Before log_prob: action_dist logits device = {action_dist.logits.device}")
                    print(f"Debug: Step {step}, Before log_prob: action_tensor device = {action_tensor.device}")
                    print(f"Debug: Step {step}, Before log_prob: action_dist logits dtype = {action_dist.logits.dtype}")
                    print(f"Debug: Step {step}, Before log_prob: action_tensor dtype = {action_tensor.dtype}")
                    print(f"Debug: Step {step}, Sampled Action Tensor: {action_tensor.item()}")
                log_prob = action_dist.log_prob(action_tensor) 
                entropy_val = action_dist.entropy()
                if self._enable_debugging_prints:
                    print(f"Debug: Step {step}, After log_prob: log_prob shape = {log_prob.shape}")
                    print(f"Debug: Step {step}, After entropy: entropy_val shape = {entropy_val.shape}") 
                values.append(value_estimate) 
                log_probs.append(log_prob.unsqueeze(-1)) 
                entropies.append(entropy_val.unsqueeze(-1)) 
                if self._enable_debugging_prints:
                    print(f"Debug: Step {step}, Before action_td creation: type(self.env.action_spec) = {type(self.env.action_spec)}")
                    if isinstance(self.env.full_action_spec, Composite):
                        print(f"Debug: Step {step}, Before action_td creation: self.env.action_spec keys = {list(self.env.full_action_spec.keys())}")
                        if "action" in self.env.full_action_spec.keys():
                            action_spec_inner = self.env.full_action_spec["action"]
                            print(f"Debug: Step {step}, Before action_td creation: self.env.action_spec['action'] type = {type(action_spec_inner)}")
                            if isinstance(action_spec_inner, TensorSpec):
                                print(f"Debug: Step {step}, Before action_td creation: self.env.action_spec['action'] shape = {action_spec_inner.shape}")
                                print(f"Debug: Step {step}, Before action_td creation: self.env.action_spec['action'] dtype = {action_spec_inner.dtype}")
                        else:
                            print("Debug: Step {step}, Before action_td creation: 'action' key not found in self.env.action_spec Composite.")
                    else:
                        if isinstance(self.env.action_spec, TensorSpec):
                            print(f"Debug: Step {step}, Before action_td creation: self.env.action_spec shape = {self.env.action_spec.shape}")
                            print(f"Debug: Step {step}, Before action_td creation: self.env.action_spec dtype = {self.env.action_spec.dtype}")
                        else:
                            print(f"Debug: Step {step}, Before action_td creation: self.env.action_spec is not a TensorSpec.")
                if not isinstance(self.env.full_action_spec, Composite) or "action" not in self.env.full_action_spec.keys():
                     raise TypeError(
                         f"Expected env.full_action_spec to be a Composite spec with an 'action' key, "
                         f"but got type {type(self.env.full_action_spec)} with keys "
                         f"{list(self.env.full_action_spec.keys()) if isinstance(self.env.full_action_spec, Composite) else 'N/A'}. "
                         f"Ensure  SubstratumBridge wraps the action space in a Composite spec."
                     )
                action_td = self.env.full_action_spec.zero() 
                if self._enable_debugging_prints:
                    print(f"Debug: Step {step}, After action_td creation: type(action_td) = {type(action_td)}")
                    print(f"Debug: Step {step}, After action_td creation: action_td = {action_td}")
                    print(f"Debug: Step {step}, After action_td creation: action_td batch_size = {action_td.batch_size}") 
                    print(f"Debug: Step {step}, After action_td creation: type(action_tensor) = {type(action_tensor)}")
                    print(f"Debug: Step {step}, After action_td creation: action_tensor shape = {action_tensor.shape}")
                
                action_td["action"] = action_tensor
                next_td = self.env.step(action_td) 
                if self._enable_debugging_prints:
                    print(f"Debug: Step {step}, After env.step: next_td batch_size = {next_td.batch_size}")
                current_td = next_td
                current_td = next_td.get("next", next_td)
                rewards.append(next_td["next"]["reward"]) 
                terminated_flags.append(next_td["next"]["terminated"])
                truncated_flags.append(next_td["next"]["truncated"])
                infos.append(next_td["next"]["info"])
                transition_td = TensorDict({
                    "observation": current_td["observation"], 
                    "action": action_td["action"], 
                    "reward": next_td["next"]["reward"], 
                    "terminated": next_td["next"]["terminated"],
                    "truncated": next_td["next"]["truncated"],
                    "next_observation": next_td["next"]["observation"], 
                }, batch_size=[])
                trajectory.append(transition_td)
                if next_td["next"]["terminated"].item() or next_td["next"]["truncated"].item(): 
                    last_value = torch.tensor([[0.0]], device=self.device)
                    break 
            except IndexError as e:
                print(f"\n--- IndexError caught in train_step at step {step} ---")
                print(f"Error: {e}")
                print("Debug Info at time of error:")
                print(f"  action_logits shape: {action_logits.shape if 'action_logits' in locals() else 'Not available'}")
                print(f"  action_dist batch_shape: {action_dist.batch_shape if 'action_dist' in locals() else 'Not available'}")
                print(f"  action_dist event_shape: {action_dist.event_shape if 'action_dist' in locals() else 'Not available'}")
                print(f"  action_tensor shape: {action_tensor.shape if 'action_tensor' in locals() else 'Not available'}")
                print(f"  action_dist logits device: {action_dist.logits.device if 'action_dist' in locals() else 'Not available'}")
                print(f"  action_tensor device: {action_tensor.device if 'action_tensor' in locals() else 'Not available'}")
                print(f"  action_dist logits dtype: {action_dist.logits.dtype if 'action_dist' in locals() else 'Not available'}")
                print(f"  action_tensor dtype: {action_tensor.dtype if 'action_tensor' in locals() else 'Not available'}")
                print(f"  action_logits (first 5 elements): {action_logits[0, :5] if 'action_logits' in locals() and action_logits.numel() > 0 else 'Not available'}")
                print(f"  action_tensor: {action_tensor if 'action_tensor' in locals() else 'Not available'}")
                traceback.print_exc()
                return None, {"steps_in_segment": step} 
            except (AttributeError, TypeError) as e: 
                 print(f"\n--- Error caught in train_step at step {step} ---")
                 print(f"Error: {e}")
                 print("Debug Info at time of error:")
                 print(f"  type(action_td): {type(action_td) if 'action_td' in locals() else 'Not available'}")
                 print(f"  action_td: {action_td if 'action_td' in locals() else 'Not available'}")
                 print(f"  action_td batch_size: {action_td.batch_size if 'action_td' in locals() and isinstance(action_td, TensorDict) else 'Not available'}")
                 print(f"  type(action_tensor): {type(action_tensor) if 'action_tensor' in locals() else 'Not available'}")
                 print(f"  action_tensor shape: {action_tensor.shape if 'action_tensor' in locals() else 'Not available'}")
                 print(f"  type(self.env.action_spec): {type(self.env.action_spec) if 'self' in locals() and hasattr(self, 'env') else 'Not available'}")
                 print(f"  type(self.env.full_action_spec): {type(self.env.full_action_spec) if 'self' in locals() and hasattr(self.env, 'full_action_spec') else 'Not available'}")
                 print(f"  type(next_td): {type(next_td) if 'next_td' in locals() else 'Not available'}")
                 print(f"  next_td keys: {list(next_td.keys()) if 'next_td' in locals() and isinstance(next_td, TensorDict) else 'Not available'}")
                 print(f"  next_td batch_size: {next_td.batch_size if 'next_td' in locals() and isinstance(next_td, TensorDict) else 'Not available'}")
                 traceback.print_exc()
                 return None, {"steps_in_segment": step}
            except Exception as e:
                 print(f"\n--- Unexpected Exception caught in train_step at step {step} ---")
                 print(f"Error Type: {type(e)}")
                 print(f"Error Message: {e}")
                 traceback.print_exc()
                 return None, {"steps_in_segment": step} 
        if not trajectory:
            return current_td, {}
        if self._enable_debugging_prints:
            print(f"Debug: Before torch.stack: len(trajectory) = {len(trajectory)}")
            if trajectory:
                print(f"Debug: Before torch.stack: trajectory[0] batch_size = {trajectory[0].batch_size}")
        trajectory_td = torch.stack(trajectory, dim=0)
        if self._enable_debugging_prints:
            print(f"Debug: After torch.stack: trajectory_td batch_size = {trajectory_td.batch_size}")
        values = torch.cat(values, dim=0) 
        log_probs = torch.cat(log_probs, dim=0) 
        entropies = torch.cat(entropies, dim=0) 
        rewards = trajectory_td["reward"] 
        terminated_flags = trajectory_td["terminated"] 
        if not (next_td["next"]["terminated"].item() or next_td["next"]["truncated"].item()):
            with torch.no_grad(): 
                _, last_value = self.model(next_td["next"])
        else:
            last_value = torch.tensor([[0.0]], device=self.device)
        returns = self.calculate_returns(rewards, terminated_flags, last_value) 
        advantage = returns - values.detach()
        policy_loss = -(log_probs * advantage).mean() 
        value_loss = F.mse_loss(values, returns.detach())
        entropy = entropies.mean() 
        total_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
        self.optimizer.zero_grad() 
        total_loss.backward() 
        self.optimizer.step()
        metrics = {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "reward_sum": rewards.sum().item(),
            "steps_in_segment": len(trajectory) 
        }
        return next_td["next"], metrics


# Experiment Runner for Graph RL
class GraphRLExperimentRunner:
    """Runs RL experiments with a graph environment and A2C model, logging results."""
    def __init__(
        self,
        initial_nx_graph: nx.DiGraph,
        graph_name: str, 
        start_node_name: Optional[Any] = None,
        terminal_node_name: Optional[Any] = None,
        results_filename: str = "experiment_results.csv",
        device: Optional[torch.device] = None,
        model_output_dim: int = 64, 
        gamma: float = 0.99,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        n_steps: int = 1,
        learning_rate: float = 0.001,
        enable_debugging_prints:bool = False 
    ):
        """Initializes the experiment runner."""
        if device is None: 
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.initial_nx_graph = initial_nx_graph
        self.graph_name = graph_name
        self.start_node_name = start_node_name
        self.terminal_node_name = terminal_node_name
        self.results_filename = results_filename
        self.model_output_dim = model_output_dim
        self.gamma = gamma 
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.n_steps = n_steps
        self.learning_rate = learning_rate
        self.gym_graph_space = convert_nx_digraph_to_gym_graph(initial_nx_graph) 
        self._enable_debugging_prints = enable_debugging_prints

    def _write_csv_header(self, num_runs: int, total_training_steps: int):
        """Writes the header row to the results CSV file if it doesn't exist."""
        header = [
            "Timestamp", "Duration_sec", "Model_Output_Dim", "Steps_Completed", "RL_Algorithm",
            "Graph_Name", 
            "Gamma", "Value_Loss_Coef","Entropy_Coef",
            "Start_Node", "Terminal_Node", "Seed", "Crashed",
            "Crash_Message", "Tot_Episode_Reward", "Avg_Episode_Reward", "Avg_Steps_Per_Episode",
            "Num_Experiment_Runs", "Total_Experiment_Steps"
        ]
        if not os.path.exists(self.results_filename):
            with open(self.results_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)

    def _append_result(self, result_data: Dict[str, Any], num_runs: int, total_training_steps: int):
        """Appends a single experiment run result to the CSV file."""
        fieldnames = [
            "Timestamp", "Duration_sec", "Model_Output_Dim","Steps_Completed", "RL_Algorithm",
            "Graph_Name", "Gamma", "Value_Loss_Coef","Entropy_Coef","Start_Node", "Terminal_Node", "Seed", "Crashed",
            "Crash_Message", "Tot_Episode_Reward", "Avg_Episode_Reward", "Avg_Steps_Per_Episode",
            "Num_Experiment_Runs", "Total_Experiment_Steps"
        ]
        result_data["Num_Experiment_Runs"] = num_runs
        result_data["Total_Experiment_Steps"] = total_training_steps
        result_data["Gamma"] = self.gamma
        result_data["Value_Loss_Coef"] = self.value_loss_coef
        result_data["Entropy_Coef"] = self.entropy_coef
        with open(self.results_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(result_data)

    def run_experiment(self, num_runs: int, total_training_steps: int):
        """Runs the specified number of experiment runs, training for total_training_steps each."""
        print(f"Starting {num_runs} experiment runs, training for {total_training_steps} steps each.")
        print(f"Results will be appended to '{self.results_filename}'.")
        self._write_csv_header(num_runs, total_training_steps)

        for run_idx in tqdm(range(num_runs), desc="Overall Experiment Progress", unit="run"):
            print(f"\n--- Starting Run {run_idx + 1}/{num_runs} ---")
            start_time = time.time()
            steps_completed = 0
            crashed = False
            crash_message = ""
            seed = np.random.randint(0, 1000000) + run_idx 
            episode_rewards = [] 
            episode_lengths = []
            current_episode_reward = 0.0
            current_episode_steps = 0
            try:
                torch.manual_seed(seed)
                np.random.seed(seed)
                torchrl_env =  SubstratumBridge(
                    initial_nx_graph=self.initial_nx_graph,
                    observation_space=self.gym_graph_space,
                    start_node_name=self.start_node_name,
                    terminal_node_name=self.terminal_node_name,
                    device=self.device,
                    enable_debug_print=False)
                if self._enable_debugging_prints:
                    print(f"Debug: Runner after creating env: type(torchrl_env.action_spec) = {type(torchrl_env.action_spec)}")
                    print(f"Debug: Runner after creating env: type(torchrl_env.full_action_spec) = {type(torchrl_env.full_action_spec)}")
                    if isinstance(torchrl_env.full_action_spec, Composite):
                        print(f"Debug: Runner after creating env: torchrl_env.full_action_spec keys = {list(torchrl_env.full_action_spec.keys())}")
                        if "action" in torchrl_env.full_action_spec.keys():
                            action_spec_inner_runner = torchrl_env.full_action_spec["action"]
                            print(f"Debug: Runner after creating env: torchrl_env.full_action_spec['action'] type = {type(action_spec_inner_runner)}")
                            if isinstance(action_spec_inner_runner, TensorSpec):
                                print(f"Debug: Runner after creating env: torchrl_env.full_action_spec['action'] shape = {action_spec_inner_runner.shape}")
                                print(f"Debug: Runner after creating env: torchrl_env.full_action_spec['action'] dtype = {action_spec_inner_runner.dtype}")
                graph_a2c_model = GraphA2C(
                    observation_spec=torchrl_env.observation_spec, 
                    action_spec=torchrl_env.action_spec,
                    output_dim=self.model_output_dim
                ).to(self.device)
                optimizer = optim.Adam(graph_a2c_model.parameters(), lr=self.learning_rate)
                trainer = GraphA2CTrainer(
                    model=graph_a2c_model,
                    env=torchrl_env,
                    optimizer=optimizer,
                    gamma=self.gamma,
                    value_loss_coef=self.value_loss_coef,
                    entropy_coef=self.entropy_coef,
                    n_steps=self.n_steps,
                    device=self.device,
                    enable_debugging_prints=False
                )
                print(f"Run {run_idx + 1}: Initializing environment and model with seed {seed}.")
                current_td = torchrl_env.reset(seed=seed) 
                steps_in_run = 0
                with tqdm(total=total_training_steps, desc=f"Run {run_idx + 1} Training Steps", unit="step", leave=False) as pbar:
                    while steps_in_run < total_training_steps:
                        next_td, metrics = trainer.train_step(current_td) 
                        if next_td is None or next_td.batch_size is None: 
                            print(f"Run {run_idx + 1} received invalid state from train_step, stopping run.")
                            break 
                        current_episode_reward += metrics.get("reward_sum", 0.0) 
                        steps_in_segment = metrics.get("steps_in_segment", 0)
                        current_episode_steps += steps_in_segment 
                        current_td = next_td 
                        steps_in_run += steps_in_segment
                        steps_completed = steps_in_run 
                        pbar.update(steps_in_segment) 
                        if current_td["terminated"].item() or current_td["truncated"].item(): 
                            episode_rewards.append(current_episode_reward) 
                            episode_lengths.append(current_episode_steps)
                            current_episode_reward = 0.0
                            current_episode_steps = 0
                            current_td = torchrl_env.reset(seed=seed + steps_in_run) 
                    print(f"Run {run_idx + 1} completed {total_training_steps} training steps.")
            except Exception as e:
                crashed = True
                crash_message = str(e)
                print(f"Run {run_idx + 1} crashed at step {steps_completed} with error: {e}")
                traceback.print_exc() 
            finally:
                avg_episode_reward = np.mean(episode_rewards) if episode_rewards else 0.0 
                total_epidsode_reward = np.sum(episode_rewards) if episode_rewards else 0.0
                avg_steps_per_episode = np.mean(episode_lengths) if episode_lengths else 0.0
                end_time = time.time()
                duration_sec = end_time - start_time
                result_data = { 
                    "Timestamp": datetime.datetime.now().isoformat(),
                    "Duration_sec": duration_sec,
                    "Model_Output_Dim": model_output_dim,
                    "Steps_Completed": steps_completed,
                    "RL_Algorithm": GraphA2C.__name__ if 'GraphA2C' in globals() else "GraphA2C",
                    "Graph_Name": self.graph_name,
                    "Gamma": self.gamma,
                    "Value_Loss_Coef": self.value_loss_coef,
                    "Entropy_Coef": self.entropy_coef,
                    "Start_Node": str(self.start_node_name),
                    "Terminal_Node": str(self.terminal_node_name),
                    "Seed": seed,
                    "Crashed": crashed,
                    "Crash_Message": crash_message,
                    "Avg_Episode_Reward": avg_episode_reward,
                    "Tot_Episode_Reward": total_epidsode_reward,
                    "Avg_Steps_Per_Episode": avg_steps_per_episode
                }
                self._append_result(result_data, num_runs, total_training_steps)
                print(f"Run {run_idx + 1} finished. Results logged. Crashed: {crashed}, Steps: {steps_completed}, Duration: {duration_sec:.2f}s, Avg Reward: {avg_episode_reward:.2f}, Avg Steps: {avg_steps_per_episode:.2f}")
        print(f"\nFinished all {num_runs} experiment runs. Results are in '{self.results_filename}'.")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
csv_log_path = r"INSERT PATH TO CSV OR JSON SYSMON FILE"
cerberus_gym_graph = graph_construction(csv_log_path, from_json=False) 
cerberus_observation_space = convert_nx_digraph_to_gym_graph(cerberus_gym_graph)
node_to_index = {node: index for index, node in enumerate(cerberus_gym_graph.nodes())}
index_to_node = {index: node for node, index in node_to_index.items()}
start_idx = np.random.random_integers(0,cerberus_gym_graph.number_of_nodes())
terminal_idx = np.random.random_integers(0, cerberus_gym_graph.number_of_nodes())
start_node_name = index_to_node[start_idx]
terminal_node_name = index_to_node[terminal_idx]
print(f"\nStart node '{start_node_name}', index: {start_idx}\nTerminal node '{terminal_node_name}', index: {terminal_idx}")


# Define training hyperparameters
trainer_gamma = 0.99
trainer_value_loss_coef = 1.0
trainer_entropy_coef = 1.0
trainer_n_steps = 10 
trainer_learning_rate = 0.01
model_output_dim = 64
results_file = "LOGGING-DEBUGGING FILE NAME AND PATH.txt"
experiment_graph_name = "GRAPH NAME"
num_runs_options = [5] 
total_training_steps_options = [100]


for num_runs in num_runs_options:
    for total_steps in total_training_steps_options:
        print(f"\n--- Running Experiment: {num_runs} runs, {total_steps} steps ---")
        # --- Instantiate the Experiment Runner ---
        try:
            exc_info = sys.exc_info()
            runner = GraphRLExperimentRunner(
                    initial_nx_graph=cerberus_gym_graph,
                    graph_name=experiment_graph_name,
                    start_node_name=start_node_name,
                    terminal_node_name=terminal_node_name,
                    results_filename=results_file,
                    device=device,
                    model_output_dim=model_output_dim,
                    gamma=trainer_gamma,
                    value_loss_coef=trainer_value_loss_coef,
                    entropy_coef=trainer_entropy_coef,
                    n_steps=trainer_n_steps,
                    learning_rate=trainer_learning_rate
            )
            runner.run_experiment(num_runs=num_runs, total_training_steps=total_steps)
        except Exception as e:
            print(f"\nAn unexpected error occurred during experiment setup or run: {e}")
            traceback.print_exc()
