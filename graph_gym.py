import networkx as nx
import gymnasium.spaces as spaces
import numpy as np
import dateutil.parser
from typing import Dict, Tuple, Set, Any
from net_analysis import _try_convert_string_to_number


def _try_convert_string_to_number(s: str) -> Any:
    """Attempts to convert a string to a float or a numerical timestamp."""
    if not isinstance(s, str): 
        return s
    try: 
        return float(s)
    except ValueError: 
        pass
    try:
        dt = dateutil.parser.parse(s)
        return dt.timestamp()
    except (ValueError, dateutil.parser.ParserError, OSError): 
        pass
    return s


def _collect_attribute_vocabulary_and_max_len(nx_graph: nx.DiGraph) -> Tuple[Dict[str, int], int]:
    """
    Collects all unique string values from node and edge attributes
    and determines the maximum list length among all attributes.

    Args:
        nx_graph: The NetworkX DiGraph.

    Returns:
        A tuple containing:
        - A dictionary mapping unique string values to integer indices.
        - The maximum length found among all attribute lists.
    """
    vocabulary: Set[str] = set()
    max_len = 0
    # node attributes
    for _, attrs in nx_graph.nodes(data=True):
        for value in attrs.values():
            if isinstance(value, list):
                max_len = max(max_len, len(value))
                for item in value:
                    evaluated_item = _try_convert_string_to_number(item)
                    if isinstance(evaluated_item, str):
                        vocabulary.add(item)
            else:
                 evaluated_value = _try_convert_string_to_number(value)
                 if isinstance(evaluated_value, str):
                      max_len = max(max_len, 1) # Treat scalar strings as list of length 1 for max_len
                      vocabulary.add(evaluated_value)
                 elif not isinstance(evaluated_value, (int, float, np.number)):
                      str_value = str(evaluated_value)
                      if str_value: # Avoid adding empty strings from None etc.
                           max_len = max(max_len, 1)
                           vocabulary.add(str_value)
    # edge attributes
    for _, _, attrs in nx_graph.edges(data=True):
        for value in attrs.values():
            if isinstance(value, list):
                max_len = max(max_len, len(value))
                for item in value:
                    evaluated_item = _try_convert_string_to_number(item)
                    if isinstance(evaluated_item, str):
                        vocabulary.add(evaluated_item)
            else:
                 evaluated_value = _try_convert_string_to_number(value)
                 if isinstance(evaluated_value, str):
                      max_len = max(max_len, 1)
                      vocabulary.add(evaluated_value)
                 elif not isinstance(evaluated_value, (int, float, np.number)):
                      str_value = str(evaluated_value)
                      if str_value:
                           max_len = max(max_len, 1)
                           vocabulary.add(str_value)
    # Create a mapping from vocabulary strings to integers
    vocab_mapping = {word: i for i, word in enumerate(sorted(list(vocabulary)))}
    return vocab_mapping, max_len


def convert_nx_digraph_to_gym_graph(nx_graph: nx.DiGraph) -> spaces.Graph:
    """
    Converts a NetworkX DiGraph to a gymnasium.spaces.Graph.

    Handles node and edge attributes that are lists containing mixed string
    and integer values by encoding strings and padding lists.
    
    NOTE: The actual graph data (attribute values, links) is not stored in the *space* object,
    only the definition of the space. When you get an observation from an environment
    that uses this space, the observation will contain the actual data, encoded
    and padded according to the space definition.

    Args:
        nx_graph: The NetworkX DiGraph to convert.

    Returns:
        A gymnasium.spaces.Graph representation of the input graph.
    """
    # build vocabulary
    _, max_list_len = _collect_attribute_vocabulary_and_max_len(nx_graph)
    effective_max_len = max(1, max_list_len)
    all_node_attr_keys = sorted(list(set().union(*(d.keys() for _, d in nx_graph.nodes(data=True)))))
    all_edge_attr_keys = sorted(list(set().union(*(d.keys() for _, _, d in nx_graph.edges(data=True)))))
    node_attr_vector_shape = len(all_node_attr_keys) * effective_max_len
    edge_attr_vector_shape = len(all_edge_attr_keys) * effective_max_len
    node_space = spaces.Box(low=-np.inf, high=np.inf, # used for the attributes (features) associated with the nodes and edges, not for the traversal process itself.
                            shape=(node_attr_vector_shape,), 
                            dtype=np.float32)
    if edge_attr_vector_shape > 0:
        edge_space = spaces.Box(low=-np.inf, 
                                high=np.inf, 
                                shape=(edge_attr_vector_shape,), 
                                dtype=np.float32)
    else:
        edge_space = None # Use None if no edge attributes are expected
    # Create the gymnasium.spaces.Graph
    gym_graph_space = spaces.Graph(
        node_space=node_space,
        edge_space=edge_space,)
    return gym_graph_space
