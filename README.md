# Substratum
![NetworkX](./images/NetworkX.png=100x200) | ![OpenAI](./images/OpenAPI.png=100x200) | ![PyTorch](./images/PyTorch.png=100x200) | ![RL](./images/reinforcement-learning.png=100x200) 

Bahirah Adewunmi | b280@umbc.edu | University of Maryland, Baltimore County
Currently under A Provisional US Patent:

This methodology models operating system state and transitions a graph, derived from open-source System Monitor (Sysmon) logs that were generated, in part, by MITRE Caldera, an autonomous adversary emulation platform. To address the variety in system event types, fields, and log formats, a mechanism was developed to capture and model parent-child processes from Sysmon logs. An OpenAI Gymnasium environment (`SubstratumGraphEnv`) was constructed to establish the perceptible basis for an RL environment. To surmount TorchRL compatibility limitations, a customized PyTorch interface was also built (SubstratumBridge) to translate gymnasium object into the PyTorch framework, enabling composite RL agent observations and discrete actions executed on the graphs. Graph Convolutional Networks (GCNs) concretize the graph’s local and global state, which feed the distinct policy and critic heads of an Advantage Actor-Critic (A2C) model. 

A white paper detailing this work has been submitted to the NeurIPS 2025 Datasets & Benchmarks Track.

The project utilizes PyTorch, TorchRL, Gymnasium, NetworkX, and Torch Geometric.

## Project Structure
- `rl_experiment.py`: Contains the main experiment runner (`GraphRLExperimentRunner`), the Graph A2C model (`GraphA2C`), and the training logic (`GraphA2CTrainer`). This is the primary script to run experiments.
- `torch_rl_env.py`: Defines a custom Gymnasium environment (`SubstratumGraphEnv`) that represents the system log graph and a TorchRL wrapper (`SubstratumBridge`) to interface it with the TorchRL library.
- `normalize_logs.py`: Provides functions to parse and normalize Sysmon logs (in both JSON and CSV formats) into a structured dictionary representing relationships between entities (processes, files, network connections, registry keys).
- `net_analysis.py`: Contains utilities for constructing a NetworkX graph from the normalized log data, performing basic graph analysis, and visualizing the graph or its subgraphs.
- `graph_gym.py`: Includes functions to convert a NetworkX graph into a Gymnasium graph space definition, handling attribute encoding and padding for compatibility with RL environments.

## Installation
1. Clone the repository.
2. Ensure you have Python 3.8+ installed.
3. Install the required libraries using the `requirements.txt` file to ensure interoperability between the libraries. It's recommended to use a virtual environment such as [virtualenv](https://virtualenv.pypa.io/en/latest/) or [Anaconda](https://www.anaconda.com/download).
```
pip install -r requirements.txt
```
4. Install `torch_geometric`. This library requires specific installation steps depending on your CUDA availability and PyTorch version. Refer to the official PyTorch Geometric installation guide. A typical CPU-only installation might look like:
```
pip install torch_geometric
```

If you have a CUDA-enabled GPU, follow the instructions for your specific CUDA version.

## Usage
1. Prepare your log data: Place your Sysmon log files (CSV or JSON) in a designated directory (e.g., data/). Update the `csv_file_path` or `sysmon_log_path` variables in `normalize_logs.py` and `rl_experiment.py` accordingly.
2. Run an experiment: Execute the `rl_experiment.py` script. This script will:
  - Load and process the specified log file using functions from `normalize_logs.py` and `net_analysis.py` to build a NetworkX graph.
  - The code converts the NetworkX graph into a Gymnasium graph space using `convert_nx_digraph_to_gym_graph` and  `graph_gym.py`.
  - Initializes the `SubstratumGraphEnv` and wrap it with `SubstratumBridge`.
  - Initializes the `GraphA2C` model and `GraphA2CTrainer`.
  - Runs the specified number of training runs for a given number of steps.
  - Log experiment results (loss, reward, etc.) to a CSV file.

You can configure experiment parameters (number of runs, training steps, model hyperparameters, log file paths, start/terminal nodes) by modifying the variables in the main execution block of rl_experiment.py.
```
python rl_experiment.py
```
3. Graph Analysis and Visualization: The `net_analysis.py` script contains commented-out examples of how to load a graph, perform analysis (`graph_analysis`), and visualize it (`draw_graph_with_proportional_size`, `downselect_visualize`). You can uncomment and run these sections independently to inspect your graph data.

## Key Components
- `normalize_logs.py`: Functions for parsing raw Sysmon logs, extracting relevant fields (process IDs, image paths, network details, registry keys, timestamps), cleaning and normalizing data, and structuring it into a dictionary representing potential graph relationships. Handles different Sysmon Event IDs. 
- `net_analysis.py`: Builds a NetworkX `DiGraph` from the normalized relationships. Nodes typically represent entities (processes, files, IPs, registry keys), and edges represent observed interactions or relationships from the logs. Includes functions for calculating graph metrics (density, centrality, clustering) and rendering graph visualizations.
- `graph_gym.py`: Bridges NetworkX and Gymnasium. It analyzes graph attributes to determine the structure and data types needed for the Gymnasium `spaces.Graph` definition, including handling variable-length list attributes and encoding strings.

- `torch_rl_env.py`:
  - `SubstratumGraphEnv`: A custom Gymnasium environment where the agent's state is its current node in the graph, and actions are selecting a target node. Rewards are based on edge weights (e.g., frequency of interaction, or potentially other metrics). The environment handles transitions based on graph adjacency and defines termination conditions (e.g., reaching a terminal node).
  - `SubstratumBridge`: A wrapper class that adapts the `SubstratumGraphEnv` (a standard Gymnasium env) to the TorchRL `EnvBase` interface, managing observation and action spaces as TorchRL Specs and handling data conversion to TensorDicts.
- `rl_experiment.py`:
  - `GraphA2C`: A PyTorch `nn.Module` implementing an Actor-Critic architecture. It uses `torch_geometric.nn.GCNConv` layers to process the graph observation (node features and edge connectivity) and outputs action logits (policy) and a state value estimate (critic).
  - `GraphA2CTrainer`: Implements the Advantage Actor-Critic (A2C) algorithm. It collects trajectories of experience from the `SubstratumBridge` environment, calculates returns and advantages, computes policy and value losses (with entropy regularization), and performs model updates using an optimizer.
  - `GraphRLExperimentRunner`: Orchestrates the entire experiment. It initializes the graph, environment, model, and trainer, runs multiple training runs, handles environment resets, calls the trainer's `train_step`, and logs results.

## Data Format
The project is designed to process Sysmon logs. The `normalize_logs.py` script currently supports:
- JSON Lines: Expected format is one JSON object per line, typically from tools like `sysmon-logstash`. It looks for specific nested keys (data_model.fields) and event codes (1, 3, 5).
- CSV: Expected format is a standard CSV output from Sysmon, with columns like `UtcTime`, `ProcessGuid`, `ProcessId`, `Image`, `ParentProcessGuid`, `ParentProcessId`, `ParentImage`, `User`, `Hostname`, `EventID`, `TargetFilename`, `TargetObject`, `RuleName`, `Protocol`, `SourceIp`, `SourcePort`, `DestinationIp`, `DestinationPortName`.

The `normalize_logs.py` script extracts and correlates the parent-child relationships and attributes from these formats. Ensure your log files have the expected structure and key names for proper parsing.

## Experiment Configuration
The `rl_experiment.py` script is configured with the following parameters for the experiment runs:
-  `csv_log_path`: Path to JSON or CSV sysmon log
- `trainer_gamma`: 0.99 (Discount factor for the RL agent)
- `trainer_value_loss_coef`: 1.0 (Coefficient for the value function loss)
- `trainer_entropy_coef`: 1.0 (Coefficient for the policy entropy term)
- `trainer_n_steps`: 10 (Number of steps collected before each model update)
- `trainer_learning_rate`: 0.01 (Learning rate for the optimizer)
- `model_output_dim`: 75 (Output dimension for the graph feature extractor in the model)
- `results_file`: Filename for saving experiment results
- `experiment_graph_name`: Name assigned to the graph for logging
- `num_runs_options`: [5] (A list of options for the number of experiment runs; currently set to 5 runs)
- `total_training_steps_options`: [100] (A list of options for the total training steps per run; Default set to 100 steps)

## Citations
The design of Substratum was verified using open-source Sysmon logs generated, in part, by MITRE Caldera. Please review the data source licensing prior to use.
- Kemmerer, Mike, and Craig Wampler. (2017) 2018. “Mitre/Brawl-Public-Game-001.” The MITRE Corporation. https://github.com/mitre/brawl-public-game-001. (Used sysmon-brawl_public_game_001.JSON)
- Pratomo, Baskoro Adi. (2023) 2023. “Bazz-066/Cerberus-Trace.” https://github.com/bazz-066/cerberus-trace. (Used NLME.csv)
  - [!IMPORTANT]: Before running `rl_experiment.py` on the BRAWL dataset, rename the following columns:  `ParentProcessId` -> `ppid` and `ProcessId` -> pid to enable `normalize_logs.process_relations_csv` to properly read in process ID fields.
