# -*- coding: utf-8 -*-
import networkx as nx
import matplotlib.pyplot as plt
import dateutil.parser
from typing import Any
from normalize_logs import process_relations_csv, process_relations_json
from networkx.drawing.nx_agraph import graphviz_layout

def clean_dicts(my_dict, enable_debugging_print:bool = False) -> dict:
    """
    Cleans the input dictionaries by removing any empty values.
    """
    keys_to_remove = []
    for key, value in my_dict.items():
        if not value:  # Checks if the dictionary is empty
            keys_to_remove.append(key)
    if enable_debugging_print:
        print("Removed", len(keys_to_remove), "empty dictionaries")
    for key in keys_to_remove:
        if enable_debugging_print:
            print(key, "is empty")
        my_dict.pop(key)
    
    return my_dict


def get_weight_attribute_types(graph:nx.Graph) -> set:
    """
    Gets all unique data types of the 'weight' attribute in a NetworkX graph.

    Args:
        graph: A NetworkX graph object.

    Returns:
        A set containing the unique data types of the 'weight' attribute found in the graph's edges.
    """
    weight_types = set()
    for u, v, data in graph.edges(data=True):
        if 'weight' in data:
            weight_types.add(type(data['weight']))
    return weight_types


def reconstruct_graph(graph:nx.Graph, traces_network:dict) -> nx.DiGraph:
    """
    Reconstructs the graph from the given parameters. The keys of the outermost 
    dictionary become nodes in the NetworkX graph. These are treated as the starting 
    points or "parents" in the context of the dict.  The keys of the inner 
    dictionaries become the nodes that are connected to the 
    parent nodes. These are the "children" in the dict.
    """
    for parent_node, connections in traces_network.items():
        if parent_node in graph.nodes:
            # Assuming attributes like parent_path, parent_type, hostname, user
            # are consistent for the 'parent' node
            if connections:
                first_neighbor_data = next(iter(connections.values()))
                if 'parent_path' in first_neighbor_data:
                    graph.nodes[parent_node]['path'] = first_neighbor_data['parent_path']
                if 'parent_type' in first_neighbor_data:
                    graph.nodes[parent_node]['type'] = first_neighbor_data['parent_type']
                if 'hostname' in first_neighbor_data:
                    graph.nodes[parent_node]['hostname'] = first_neighbor_data['hostname']
                if 'user' in first_neighbor_data:
                    graph.nodes[parent_node]['user'] = first_neighbor_data['user']
                if 'ppid' in first_neighbor_data:
                    graph.nodes[parent_node]['ppid'] = first_neighbor_data['ppid']
                if 'command_line' in first_neighbor_data:
                    graph.nodes[parent_node]['command_line'] = first_neighbor_data['command_line']
        for child_node, attributes in connections.items():
            if child_node not in graph.nodes:
                graph.add_node(child_node)
            weight = None
            if 'weight' in attributes and isinstance(attributes['weight'], list) and attributes['weight']:
                try:
                    weight = int(attributes['weight'][0])
                except (ValueError, TypeError):
                    print(f"Warning: Could not convert weight '{attributes['weight'][0]}' to integer for edge ({parent_node}, {child_node}). Skipping weight for this edge.")

            if weight is not None:
                graph.add_edge(parent_node, child_node, weight=weight)
            else:
                graph.add_edge(parent_node, child_node) 
            if 'child_path' in attributes:
                graph.nodes[child_node]['path'] = attributes['child_path']
            if 'child_type' in attributes:
                graph.nodes[child_node]['type'] = attributes['child_type']
            if 'pid' in attributes:
                graph.nodes[child_node]['pid'] = attributes['pid']
            if 'registry_hive' in attributes:
                graph.nodes[child_node]['registry_hive'] = attributes['registry_hive']
            if 'end_time' in attributes:
                graph.nodes[child_node]['end_time'] = attributes['end_time']
            if 'start_time' in attributes:
                graph.nodes[child_node]['start_time'] = attributes['start_time']
            if 'duration_sec' in attributes:
                graph.nodes[child_node]['duration_sec'] = attributes['duration_sec']
            if 'duration_min' in attributes:
                graph.nodes[child_node]['duration_min'] = attributes['duration_min']
            if 'transport' in attributes:
                graph.nodes[child_node]['transport'] = attributes['transport']
            if 'src_port' in attributes:
                graph.nodes[child_node]['src_port'] = attributes['src_port']
            if 'src_p_name' in attributes:
                graph.nodes[child_node]['src_p_name'] = attributes['src_p_name']
            if 'rule' in attributes:
                graph.nodes[child_node]['rule'] = attributes['rule'][0]
            if 'event_type' in attributes:
                graph.nodes[child_node]['event_type'] = attributes['event_type']
    return graph


def graph_construction(sysmon_log_path:str, from_json:bool=True) -> nx.DiGraph:
    """
    Constructs a graph from the given Sysmon log path.
    """
    if from_json:
       network = process_relations_json(sysmon_log_path)
    else:
        network = process_relations_csv(sysmon_log_path)
    clean_network = clean_dicts(network)
    unstruct_G = nx.from_dict_of_dicts(clean_network, create_using=nx.DiGraph)
    graph = reconstruct_graph(unstruct_G, network)
    return graph


def downselect_visualize(digraph: nx.DiGraph) -> None:
    """
    Downselects and visualizes the largest possible subgraph of a DiGraph
    where interconnected nodes each have multiple child nodes.

    Args:
        digraph: A NetworkX DiGraph.
    """
    candidate_nodes = [node for node in digraph.nodes() if digraph.out_degree(node) > 1] 
    if not candidate_nodes:
        print("No nodes found with multiple child nodes.")
        return
    candidate_subgraph = digraph.subgraph(candidate_nodes) 
    weakly_connected_components_list = list(nx.weakly_connected_components(candidate_subgraph)) 
    if not weakly_connected_components_list:
        print("No interconnected nodes found with multiple child nodes.")
        return
    valid_components = [] 
    for component in weakly_connected_components_list:
        is_valid = True
        for node in component:
            if digraph.out_degree(node) <= 1:
                is_valid = False
                break
        if is_valid:
            valid_components.append(component)
    if not valid_components:
        print("No interconnected subgraph found where all nodes have multiple child nodes.")
        return
    largest_component = max(valid_components, key=len) # Select the largest valid component
    subgraph_to_visualize = digraph.subgraph(largest_component) # Create the subgraph to visualize from the original DiGraph
    edge_widths = [] # Determine edge widths based on weight attribute
    for u, v, data in subgraph_to_visualize.edges(data=True):
        weight = data.get('weight', 0.01)  # Default weight to .01 if not present
        edge_widths.append(weight * 0.01) 
    if subgraph_to_visualize.nodes():   
            try:
                node_sizes = [digraph.out_degree(node) * 1 for node in subgraph_to_visualize.nodes()]
                pos = graphviz_layout(subgraph_to_visualize, prog='neato', args='-Gsep=2.0')  # You can try other programs like 'neato', 'fdp'
                nx.draw_networkx_nodes(subgraph_to_visualize, pos, node_size=node_sizes, node_color='#aec7e8', alpha=0.5)
                nx.draw_networkx_edges(subgraph_to_visualize, pos, width=edge_widths, alpha=0.4, edge_color="#3c3f3f")
                nx.draw_networkx_labels(subgraph_to_visualize, pos, font_size=10, verticalalignment='top',font_weight="bold", font_color='#000000', font_family='serif')
                plt.axis('off')
                plt.show()
            except Exception as e:
                print(f"Error with Graphviz layout: {e}")
                # Fallback to a NetworkX layout if Graphviz fails
                pos = nx.spring_layout(subgraph_to_visualize, seed=42)
                nx.draw(subgraph_to_visualize, pos, with_labels=True, k=0.3, node_size=node_sizes,
                        node_color="lightblue", font_size=5, arrowsize=10)
                plt.title("Largest Subgraph with Interconnected Nodes (Multiple Children)")
                plt.show()


def draw_graph_with_proportional_size(graph: nx.Graph) -> None:
    """
    Draws a NetworkX graph using graphviz_layout, with node size proportional
    to the number of children and edge width proportional to the weight.

    Args:
        graph: A NetworkX DiGraph (recommended for the concept of children)
               or a Graph where neighbors are considered children.
    """
    node_sizes = [] # Determine node sizes based on the number of children (out-degree for DiGraph)
    if isinstance(graph, nx.DiGraph):
        node_sizes = [graph.out_degree(node) * 5 for node in graph.nodes()]
    else:
        node_sizes = [len(list(graph.neighbors(node))) * 5 for node in graph.nodes()] 
    edge_widths = [] 
    for _, _, data in graph.edges(data=True):
        weight = data.get('weight', 0.005) 
        edge_widths.append(weight * 0.005)  
    try:
        pos = graphviz_layout(graph, prog='neato', args='-Gsep=12.0') 
        nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, node_color="#11521e", alpha=0.5)
        nx.draw_networkx_edges(graph, pos, width=edge_widths, alpha=0.6, edge_color="#a49595")
        nx.draw_networkx_labels(graph, pos, font_size=10, font_weight="bold", font_color='#000000', font_family='serif')
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Error during graph drawing: {e}")


def graph_analysis(graph: nx.Graph) -> dict:
    """
    Analyzes the graph and prints various metrics.

    Args:
        graph: A NetworkX graph object.
    """
    # Cerberus Traces Analysis
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    density = nx.density(graph)
    degree = nx.average_degree_connectivity(graph, weight="weight")
    # Cerberus Traces In Degree Centrality
    in_degree_centrality = nx.in_degree_centrality(graph)
    sorted_centrality_items = sorted(in_degree_centrality.items(), key=lambda item: item[1], reverse=True)
    sorted_in_degree_centrality = dict(sorted_centrality_items)
    # Cerberus Traces Betweenness Centrality
    betweenness_centrality = nx.betweenness_centrality(graph, weight="weight")
    sorted_betweenness_centrality_items = sorted(betweenness_centrality.items(), key=lambda item: item[1], reverse=True)
    sorted_betweeness_centrality = dict(sorted_betweenness_centrality_items)
    # Cerberus Traces Clustering Coefficient
    clustering_coefficient = nx.clustering(graph, weight="weight")
    sorted_clustering_items = sorted(clustering_coefficient.items(), key=lambda item: item[1], reverse=True) 

    print(f"Node Count: {node_count}")
    print(f"Edge Count: {edge_count}")
    print(f"Density: {density}")
    print(f"Average Degree Connectivity: {degree}")
    print(f"In-Degree Centrality: {sorted_in_degree_centrality}")
    print(f"Betweenness Centrality: {sorted_betweeness_centrality}")
    print(f"Clustering Coefficient: {sorted_clustering_items}")
    analysis = {
        "node_count": node_count,
        "edge_count": edge_count,
        "density": density,
        "degree": degree,
        "in_degree_centrality": sorted_in_degree_centrality,
        "betweenness_centrality": sorted_betweeness_centrality,
        "clustering_coefficient": sorted_clustering_items}
    return analysis


def _try_convert_string_to_number(s: str) -> Any:
    """
    Attempts to convert a string to a float or a numerical timestamp representation.
    If unsuccessful, returns the original string.

    Args:
        s: The input string.

    Returns:
        A float if the string represents a number, a numerical timestamp
        (Unix timestamp) if it represents a parseable date/time,
        otherwise the original string.
    """
    try:
        return float(s) # Try converting to float first
    except ValueError:
        pass
    try:
        dt = dateutil.parser.parse(s) # Try parsing as a timestamp and convert to Unix timestamp
        return dt.timestamp()
    except (ValueError, OSError, dateutil.parser.ParserError):
        pass
    return s


def find_longest_path_or_handle_cycles(graph: nx.DiGraph, weight: str = 'weight'):
    """
    Finds the longest path in a DAG or calculates the longest path in its condensation graph
    if the graph contains cycles. Use to find the initial and terminal nodes for RL experiment

    Args:
        graph: A NetworkX Directed Graph (may contain cycles).
        weight: The name of the edge attribute to use as the weight for path length calculation.
                Defaults to 'weight'.

    Returns:
        A tuple containing:
        - The length of the path (sum of original weights).
        - The nodes in the path. For a cyclic graph, this will be a list
          of tuples, where each tuple represents a Strongly Connected Component (SCC).
        - A string message explaining what the path represents (longest path in DAG,
          or longest path in condensation graph).
        Returns (0, [], "Graph is empty or has no edges.") if the graph is empty or has no edges.
        Returns (None, None, "Error message") if an error occurs.
    """
    if not graph or not graph.edges():
         # Check if graph has nodes but no edges - it's a DAG with no paths
         if graph.nodes():
              return 0, [], "Graph has nodes but no edges (is a DAG). No paths exist."
         # Otherwise, it's an empty graph
         return 0, [], "Graph is empty or has no edges."

    if nx.is_directed_acyclic_graph(graph):
        print("Graph is a Directed Acyclic Graph (DAG). Finding longest path directly.")
        try:
            # nx.dag_longest_path_length returns the length (sum of weights)
            length = nx.dag_longest_path_length(graph, weight=weight)
            # nx.dag_longest_path returns the nodes in the path
            path = nx.dag_longest_path(graph, weight=weight)
            return length, path, "Longest path in the original DAG."
        except nx.NetworkXNoPath:
             # This happens if the DAG has nodes but no edges, or no path between any two nodes
             return 0, [], "Graph is a DAG but has no path."
        except Exception as e:
             print(f"An error occurred while finding longest path in DAG: {e}")
             return None, None, f"Error finding longest path in DAG: {e}"
    else:
        print("Graph contains cycles. Cannot apply nx.dag_longest_path directly.")
        print("Calculating the condensation graph to find the longest path between strongly connected components.")
        # Calculate the condensation graph
        C = nx.condensation(graph)
        # We need to calculate weights for the edges in the condensation graph
        # based on the original graph's edge weights. 
        C_weighted = nx.DiGraph()
        # Add nodes from the condensation graph (these are tuples of original nodes)
        C_weighted.add_nodes_from(C.nodes(data=True))

        for u_scc, v_scc, _ in C.edges(data=True):
            u_nodes = C.nodes[u_scc]['members'] 
            v_nodes = C.nodes[v_scc]['members'] 
            total_weight_between_sccs = 0.0
            edges_contributing = [] 
            for u_orig in u_nodes:
                for v_orig in v_nodes:
                    if graph.has_edge(u_orig, v_orig):
                        edge_data = graph.get_edge_data(u_orig, v_orig)
                        original_edge_weight = edge_data.get(weight, 1.0) 
                        try:
                            numerical_weight = float(_try_convert_string_to_number(original_edge_weight))
                            total_weight_between_sccs += numerical_weight
                            edges_contributing.append((u_orig, v_orig, numerical_weight))
                        except (ValueError, TypeError):
                            print(f"Warning: Edge ({u_orig}, {v_orig}) has non-numerical or unconvertible weight '{original_edge_weight}'. Ignoring for condensation graph weight sum.")
            # Add edge to the weighted condensation graph if there's a connection
            # If total_weight_between_sccs is 0 (e.g., no edges or all had non-numerical weights)
            if total_weight_between_sccs > 0 or (total_weight_between_sccs == 0 and edges_contributing):
                 C_weighted.add_edge(u_scc, v_scc, weight=total_weight_between_sccs)
        try:
            condensation_path_sccs = nx.dag_longest_path(C_weighted, weight='weight')
            condensation_path_length = nx.dag_longest_path_length(C_weighted, weight='weight')
            # The path is a list of SCCs. We return this list of tuples.
            return condensation_path_length, condensation_path_sccs, "Longest path in the condensation graph (path of Strongly Connected Components)."
        except nx.NetworkXNoPath:
             # This can happen if the condensation graph has nodes but no edges
             return 0, [], "Condensation graph has no path."
        except Exception as e:
             print(f"An error occurred while finding longest path in condensation graph: {e}")
             return None, None, f"Error finding longest path in condensation graph: {e}"


# Cerberus Traces Graph Construction
# csv_log_path = r"data\NLME.csv"
# Cerberus_Traces_G = graph_construction(csv_log_path, from_json=False)
# Cerberus Traces Graph Analysis
# Cerberus_Traces_analysis = graph_analysis(Cerberus_Traces_G)
# Cerberus Traces Graph Visualization
# downselect_visualize(Cerberus_Traces_G)


# BRAWL Graph Construction
# json_log_path = r"data\sysmon-brawl_public_game_001.json"
# BRAWL_G = graph_construction(json_log_path, from_json=True)
# BRAWL Graph Analysis
# BRAWL_analysis = graph_analysis(BRAWL_G)
# BRAWL Graph Visualization
#draw_graph_with_proportional_size(BRAWL_G)
