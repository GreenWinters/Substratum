"""
normalize_logs.py
This script processes and normalizes Sysmon logs in both JSON and CSV formats to extract relationships 
between processes, network connections, registry operations, and file creations. The relationships are 
stored in a nested dictionary structure, capturing details such as parent-child process relationships, 
network connections, registry modifications, and file creation events.
The script includes functions for:
- Reading JSON lines and extracting unique keys from JSON objects.
- Parsing image paths and file types.
- Processing Sysmon logs for process creation, network connections, and termination events.
- Cleaning and normalizing parent file names and node IDs.
- Handling CSV-based Sysmon logs for process creation, registry operations, and file creation events.
- Managing orphan processes that lack parent process information.
Key Features:
- Handles Event IDs 1 (Process Creation), 3 (Network Connection), 5 (Process Termination), 
    11 (File Creation), 12/13 (Registry Operations), and 2 (File Process Change).
- Supports error handling for invalid JSON, missing keys, and unexpected data formats.
- Provides utilities for cleaning and extracting relevant information from file paths, registry objects, 
    and parent node IDs.
Usage:
- Update the `csv_file_path` and `sysmon_log_path` variables with the paths to your Sysmon CSV and JSON logs.
- Call `process_relations_json()` for JSON logs or `process_relations_csv()` for CSV logs to extract relationships.
Dependencies:
- Python standard libraries: json, csv, os, re, datetime, traceback
Author:
- 
Date:
- February 20, 2025
"""
import json
import csv
import os
import re
import datetime
import traceback


def read_json_lines(file_path):
    """
    Reads a file containing JSON objects, one per line, and returns a list of parsed JSON objects.

    Args:
        file_path (str): The path to the file containing JSON lines.

    Returns:
        list: A list of dictionaries, each representing a JSON object from the file.

    Raises:
        json.JSONDecodeError: If a line in the file is not valid JSON, it will be skipped and a message will be printed.
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                record = json.loads(line)
                data.append(record)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON: {line.strip()}")
    return data


def get_all_keys(json_obj, keys=None):
    """
    Recursively retrieves all unique keys from a JSON object.

    Args:
        json_obj (dict or list): The JSON object to extract keys from. It can be a dictionary or a list of dictionaries.
        keys (list, optional): A list to store the unique keys. If not provided, a new list will be created.

    Returns:
        list: A list of unique keys found in the JSON object.
    """
    if keys is None:
        keys = []
    if isinstance(json_obj, dict):
        for key, value in json_obj.items():
            if key not in keys:
                keys.append(key)
            get_all_keys(value, keys)
    elif isinstance(json_obj, list):
        for item in json_obj:
            get_all_keys(item, keys)
    return keys


def image_string_parse(row: dict, parent:bool=True):
    """
    Parses the image path and file type from a row of data.

    Args:
        row (dict): A dictionary containing the row data.

    Returns:
        tuple: A tuple containing the file path and file type.
    """
    file_path, file_type = os.path.splitext(row.get("image_path"))
    path, file = os.path.split(file_path)
    if parent:
        parent_file_path, parent_file_type = os.path.splitext(row.get("parent_image_path"))
        _ , parent_file = os.path.split(parent_file_path)
        return path, file, file_type, parent_file, parent_file_type
    return path, file, file_type, None, None


def process_creation_json(entry:dict, relationships:dict):
    """
    Extracts parent executable and child executable from a Sysmon log entry.

    Args:
        entry: A Sysmon log entry (dictionary) in JSON format.

    Returns:
        A dictionary indexed by nested keys of parent executable and the child
    """
    row = entry.get("data_model").get("fields")
    cmd = row.get("command_line")
    pid = int(row.get("pid"))
    ppid = int(row.get("ppid"))
    path, file, file_type, parent_file, parent_file_type = image_string_parse(row)
    user = row.get("user")
    hostname = row.get("hostname")
    time = row.get("utc_time") # string
    event_type = "process_creation"
    if parent_file not in relationships:
        relationships[parent_file] = {}
    if file not in relationships[parent_file]:
        relationships[parent_file][file] = {"weight":[1], "command_line":[cmd], "child_path":[path],
                                            "child_type":[file_type], "parent_type":[parent_file_type], 
                                             "event_type":[event_type], "start_time":[time]}
        relationships[parent_file][file]["pid"]= [pid] # record the varying pid that call the parent and child files
        relationships[parent_file][file]["ppid"] = [ppid]
        relationships[parent_file][file]["user"] = [user]
        relationships[parent_file][file]["hostname"] = [hostname]
        
    else:
        relationships[parent_file][file]["weight"][0] += 1 #increment the weight if the relationship already exists
        relationships[parent_file][file]["pid"].append(pid)
        pid_ls = list(set(relationships[parent_file][file]["pid"]))
        relationships[parent_file][file]["pid"] = pid_ls
        relationships[parent_file][file]["ppid"].append(ppid)
        # Remove duplicate PPIDs to ensure the list contains only unique parent process IDs
        ppid_ls = list(set(relationships[parent_file][file]["ppid"]))
        relationships[parent_file][file]["ppid"] = ppid_ls
        relationships[parent_file][file]["user"].append(user)
        user_ls = list(set(relationships[parent_file][file]["user"]))
        relationships[parent_file][file]["user"] = user_ls
        relationships[parent_file][file]["hostname"].append(hostname)
        hostname_ls = list(set(relationships[parent_file][file]["hostname"]))
        relationships[parent_file][file]["hostname"] = hostname_ls
    return relationships


def network_connection_json(entry:dict, relationships:dict):
    """
    Extracts network connection information from a Sysmon log entry. 
    NOTE fully qualified domain name (FQDN)

    Args:
        entry: A Sysmon log entry (dictionary) in JSON format.

    Returns:
        A dictionary indexed by nested keys of parent executable and the child
    """
    event_data = entry.get("data_model").get("fields")
    pid = int(event_data.get("pid"))
    user = event_data.get("user")
    file, file_type = os.path.splitext(event_data.get("image_path"))
    path, file = os.path.split(file)
    transport = event_data.get("transport")
    src_ip = event_data.get("src_ip")
    src_port = event_data.get("src_port")
    src_p_name = event_data.get("src_port_name")
    hostname = event_data.get("hostname")
    time = event_data.get("utc_time") # string
    event_type = "network_creation"
    if file not in relationships:
        relationships[file] = {}
    if src_ip not in relationships[file]:
        relationships[file][src_ip] = {"weight":[1], "child_path":[path], "file":[file],
                                    "child_type":[file_type], "transport":[transport], 
                                    "src_port":[src_port], "src_p_name":[src_p_name], 
                                    "event_type":[event_type], "start_time":[time]}
        relationships[file][src_ip]["pid"]= [pid] # record the varying pid that call the parent and child files
        relationships[file][src_ip]["user"] = [user]
        relationships[file][src_ip]["hostname"] = [hostname]
    else:
        relationships[file][src_ip]["weight"][0] += 1 #increment the weight if the relationship already exists
        relationships[file][src_ip]["pid"].append(pid)
        pid_ls = list(set(relationships[file][src_ip]["pid"]))
        relationships[file][src_ip]["pid"] = pid_ls
        relationships[file][src_ip]["user"].append(user)
        user_ls = list(set(relationships[file][src_ip]["user"]))
        relationships[file][src_ip]["user"] = user_ls
        relationships[file][src_ip]["hostname"].append(hostname)
        hostname_ls = list(set(relationships[file][src_ip]["hostname"]))
        relationships[file][src_ip]["hostname"] = hostname_ls
    return relationships    


def process_relations_json(sysmon_log_path:str):
    """
    Extracts parent process IDs (PPIDs) and process IDs (PIDs) from Sysmon logs in JSON format.
    NOTE Only Event IDs are 0,1,3 in sysmon-brawl_public_game_001.json

    Args:
        sysmon_logs: A list of Sysmon log entries (dictionaries) in JSON format.

    Returns:
        A dictionary indexed by nested keys of parent executable and the child executable.
    """
    relationships = {}
    sysmon_log = read_json_lines(sysmon_log_path)

    for entry in sysmon_log:
        event_code = entry.get("data_model").get("fields").get("event_code")
        if event_code == "1":  # EventID 1: Process creation
            try:
                relationships = process_creation_json(entry, relationships)
            except (ValueError, TypeError, AttributeError, KeyError) as e:
                print(f"Error processing log entry: {e}\n{entry}\n")
                print(traceback.format_exc())
        elif event_code == "3":  # EventID 3: Network connection
            try:
                relationships = network_connection_json(entry, relationships)
            except (ValueError, TypeError, AttributeError, KeyError) as e:
                print(f"Error processing log entry: {e}\n{entry}\n")
                print(traceback.format_exc())
        elif event_code == "5":  # EventID 5: Termination
            try:
                row = entry.get("data_model").get("fields")
                for parent_file, parent_info in relationships.items():
                    if isinstance(parent_info, dict):
                        for file, child_node_details in parent_info.items():
                            if isinstance(child_node_details, dict) and "pid" in child_node_details:
                                pid = int(row.get("pid"))
                                if  pid in child_node_details.get("pid"):
                                    relationships[parent_file][file]["end_time"] = [row.get("utc_time")]
                                    end_time = datetime.datetime.strptime(row.get("utc_time"),"%Y-%m-%d %H:%M:%S.%f")
                                    start_time = datetime.datetime.strptime(relationships[parent_file][file]["start_time"][0],"%Y-%m-%d %H:%M:%S.%f")
                                    difference = end_time - start_time
                                    relationships[parent_file][file]["duration_sec"] = [str(difference.microseconds/6000000)]
            except (ValueError, TypeError, AttributeError, KeyError) as e:
                print(f"Error processing row entry: {e}\n{row}\n\n{relationships[parent_file][file]}")
                print(traceback.format_exc())
    return relationships


def clean_parent_file(parent_file:str):
    '''
    Cleans the parent file name by removing certain prefixes or patterns.
    '''
    if 'VirusShare'in parent_file or 'virusshare' in parent_file:
        parent_file = "VirusShare"
    elif "XituuUhate" in parent_file:
        parent_file = "XituuUhate"
    elif re.match(r"^(?=.*?\d.*?\d.*?\d)[^.-]*?[.-]", parent_file):
        parent_file = re.sub(r"^(?=.*?\d.*?\d.*?\d)[^.-]*?[.-]", "", parent_file)
    elif re.match(r'^[A-Za-z]:\\Users\\\w+\\Documents\\', parent_file):
        parent_file = "Temp_Doc"
    elif re.match(r'^\{[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\} \{[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\} 0x[0-9A-Fa-f]+$', parent_file):
        parent_file = "COM_ID"
    elif re.match(r'^[0-9A-Za-z&._-]+\{[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\}$', parent_file):
        parent_file = "PnP_Device"
    else:
        return parent_file
    return parent_file


def clean_parent_node_id(full_parent_node_id: str):
    """
    Extracts and returns the parent node ID from a given string.

    Args:
        full_parent_node_id (str): The full string containing the parent node ID.

    Returns:
        str: The extracted parent node ID.

    Raises:
        ValueError: If the format of the parent node ID is invalid.
    """
    match = re.search(r"\{([0-9A-Fa-f]{8})-", full_parent_node_id)
    if match:
        parent_node_id = match.group(1)
        return parent_node_id
    else:
        raise ValueError(f"Invalid format for parent_node_id: {parent_node_id}")


def process_creation_csv(row:dict, relationships:dict, parent_node_lookup:dict):
    """
    Processes a row of process creation data and updates the process relationships and parent process ID lookup. 
    Unlike JSON, csv files do not have a parent process ID in the same row as the child process. Map child processes using parent_node_ids and ppid==pid
    ppid are assigned to processes that later call the child process. PPID -> PID relationship

    NOTE NetworkX needs consistent metadata field file formats to be able to reconstruct the graph. Hence why one field values are a list
    
    Args:
        row (dict): A dictionary containing process creation data with keys such as "ppid", "pid", "product", 
                    "user", "image_path", and "hostname".
        relationships (dict): A dictionary to store the relationships between processes.
        parent_node_lookup (dict, optional): A dictionary to store parent process information. Defaults to {}.
    
    Returns:
        tuple: A tuple containing the updated relationships and ppid_lookup dictionaries.
    """
    ppid = int(row.get("ppid"))
    pid = int(row.get("pid"))
    user = row.get("User")
    time = row.get("UtcTime") # string
    file_path, file_type = os.path.splitext(row.get("Image"))
    path, file = os.path.split(file_path)
    file = clean_parent_file(file)
    hostname = row.get("host_name")
    full_parent_node_ids = row.get("parent_node_id")
    parent_node_id = clean_parent_node_id(full_parent_node_ids)
    event_type = "process_creation"
    if parent_node_id not in parent_node_lookup.keys(): # Ensure parent_node_id exists in the lookup table
        parent_node_lookup[parent_node_id] = {}
    if pid not in parent_node_lookup[parent_node_id].keys(): # If the pid is not in the lookup table, add it
        parent_node_lookup[parent_node_id][pid] = {"parent_path":[path], "parent_type":[file_type], 
                                                     "parent_file":[file], "ppid":[ppid], "user":[user], 
                                                     "event_type":[event_type],"hostname":[hostname], 
                                                     "start_time":[time],"end_time":list(),"duration_min":list()}
    if ppid not in parent_node_lookup[parent_node_id].keys(): # If the ppid is not in the lookup table, add it
        parent_node_lookup[parent_node_id][ppid] = {"parent_path":[path], "parent_type":[file_type],
                                                    "parent_file":[file], "pid":[pid], "user":[user], 
                                                    "event_type":[event_type],"hostname":[hostname], 
                                                    "start_time":[time],"end_time":list(),"duration_min":list()}
    relationships[file] = {}
    return relationships, parent_node_lookup
    

def clean_target_object(target_object: str):
    """
    Cleans and extracts relevant parts of a registry target object string.

    Args:
        target_object (str): The registry target object string to be cleaned.

    Returns:
        tuple: A tuple containing the registry hive (if found) and the cleaned target object name.
    """
    hive_pattern = r"^(HKU|HKLM|HKCR|HKCC|HKPD)"
    bracket_pattern = r"\{[^}]*\}|\[[^\]]*\]"
    hive_match = re.search(hive_pattern, target_object)
    hive = hive_match.group(1) if hive_match else None
    cleaned_path = re.sub(bracket_pattern, "", target_object)
    path_without_hive = re.sub(hive_pattern, "", cleaned_path).lstrip("\\")
    parts = path_without_hive.split("\\")
    target_object = parts[-1].strip() if parts else None
    return hive, target_object


def registry_operations_csv(row:dict, relationships:dict):
    """
    The function extracts relevant information from the row, such as process ID, parent file path, rule name, 
    hostname, event type, and registry target object. It then updates the process relationships dictionary 
    with this information, incrementing the weight if the relationship already exists.

    NOTE NetworkX needs consistent metadata field file formats to be able to reconstruct the graph. Hence why one field values are a list

    Args:
        row (dict): A dictionary containing registry operation data with keys such as "pid", "image_path", 
                    "RuleName", "hostname", "EventType", and "TargetObject".
        relationships (dict): A dictionary to store and update process relationships.
    Returns:
        dict: Updated process relationships dictionary.
    """
    pid = int(row.get("pid"))
    parent_file_path, parent_file_type = os.path.splitext(row.get("Image"))
    parent_path, parent_file = os.path.split(parent_file_path)
    parent_file = clean_parent_file(parent_file)
    rule = row.get("RuleName")
    hostname = row.get("host_name")
    time = row.get("UtcTime") # string
    event_type = re.sub(r'SetValue|DeleteValue', lambda match: 'set_value' if match.group(0) == 'SetValue' else 'delete_value', row.get("EventType"))
    registry_hive, registry_value_name = clean_target_object(row.get("TargetObject"))
    if parent_file not in relationships:
        relationships[parent_file] = {}
    if registry_value_name not in relationships[parent_file]:
        relationships[parent_file][registry_value_name] = {"weight":[1], "parent_path":[parent_path],
                                                           "parent_type":[parent_file_type], "start_time":[time], 
                                                           "registry_hive":[registry_hive],"end_time":list(), "duration_min":list(),
                                                            "rule":[rule], "event_type":[event_type]}
        relationships[parent_file][registry_value_name]["pid"]= [pid] # record the varying pid that call the parent and child files
        relationships[parent_file][registry_value_name]["hostname"] = [hostname]
    else:
        relationships[parent_file][registry_value_name]["weight"][0] += 1 #increment the weight if the relationship already exists
        relationships[parent_file][registry_value_name]["pid"].append(pid)
        pid_ls = list(set(relationships[parent_file][registry_value_name]["pid"]))
        relationships[parent_file][registry_value_name]["pid"] = pid_ls
        relationships[parent_file][registry_value_name]["hostname"].append(hostname)
        hostname_ls = list(set(relationships[parent_file][registry_value_name]["hostname"]))
        relationships[parent_file][registry_value_name]["hostname"] = hostname_ls
    return relationships


def file_creation_change_csv(row:dict, relationships:dict, parent_node_lookup:dict, event_id:int, enable_debugging_prints:bool = False):
    """
    Processes a row of file creation data and updates the process relationships.
    NOTE There are orphan processes that are not in the parent_node_lookup table.
    NOTE NetworkX needs consistent metadata field file formats to be able to reconstruct the graph. Hence why one field values are a list

    Args:
        row (dict): A dictionary containing file creation data with keys such as "pid", "image_path", 
                    "TargetFilename", and "hostname".
        relationships (dict): A dictionary to store the relationships between processes.
    
    Returns:
        dict: Updated process relationships dictionary.
    """
    pid = int(row.get("pid"))
    file_path, file_type = os.path.splitext(row.get("TargetFilename"))
    path, file = os.path.split(file_path)
    file = clean_parent_file(file)
    hostname = row.get("host_name")
    time = row.get("UtcTime") # string
    if event_id == 11:
        event_type = "file_creation"
    else:
        event_type = "file_process_change"
    full_parent_node_id = row.get("parent_node_id")
    if full_parent_node_id is None and enable_debugging_prints:
        print(f"Skipping row due to missing parent_node_id: {row}")
        return relationships
    parent_node_id = clean_parent_node_id(full_parent_node_id)
    if parent_node_id not in parent_node_lookup.keys() or pid not in parent_node_lookup[parent_node_id].keys(): #Parent process not in the lookup table
        if enable_debugging_prints:
            print(f"Parent node id {parent_node_id} from {row.get('parent_node_id')} is not found in relationships. Skipping event type/event ID  {event_type}/{event_id}.")
        global orphan_cnt
        orphan_cnt += 1
        if "orphan" not in relationships:
            relationships["orphan"] = {}
        if file not in relationships["orphan"]:
            relationships["orphan"][file] = {"weight":[1],
                                             "pid":[pid],
                                            "hostname":[hostname],
                                            "child_path":[path], 
                                            "child_type":[file_type],
                                            "end_time":list(),
                                            "duration_min":list(), 
                                            "start_time": [time],
                                            "event_type":[event_type]}
        else:
            relationships["orphan"][file]["weight"][0] += 1
            relationships["orphan"][file]["pid"].append(pid)
            pid_ls = list(set(relationships["orphan"][file]["pid"]))
            relationships["orphan"][file]["pid"] = pid_ls
            relationships["orphan"][file]["hostname"].append(hostname)
            hostname_ls = list(set(relationships["orphan"][file]["hostname"]))
            relationships["orphan"][file]["hostname"] = hostname_ls
        return relationships
    if pid in parent_node_lookup[parent_node_id]: #If parent node id is found in the relationships
        parent_info = parent_node_lookup[parent_node_id][pid]
        parent_file_ls = parent_info['parent_file']
        for parent_file in parent_file_ls: 
            if file not in relationships[parent_file]:
            # Assumes parent file in relationship dict
                relationships[parent_file][file] = {"weight":[1],
                                                    "parent_path":parent_info['parent_path'], 
                                                    "parent_type":parent_info['parent_type'],
                                                    "hostname":[hostname],
                                                    "child_path":[path], 
                                                    "child_type":[file_type],
                                                    "parent_event_type":parent_info['event_type'], 
                                                    "end_time":list(),
                                                    "duration_min":list(), 
                                                    "start_time": [time], 
                                                    "event_type":[event_type]}
                relationships[parent_file][file]["ppid"]= parent_info['ppid'] if 'ppid' in parent_info.keys() else parent_info['pid']
                relationships[parent_file][file]["pid"]= [pid]
                relationships[parent_file][file]["parent_hostname"] = parent_info['hostname']
                relationships[parent_file][file]["user"] = parent_info['user']
            else:
                relationships[parent_file][file]["weight"][0] += 1 #increment the weight if the relationship already exists
                if 'ppid' in parent_info.keys():
                    ppid = parent_info['ppid'][0]
                else:
                    ppid = parent_info['pid'][0]
                relationships[parent_file][file]["ppid"].append(ppid)
                ppid_ls = list(set(relationships[parent_file][file]["ppid"]))
                relationships[parent_file][file]["ppid"] = ppid_ls
                relationships[parent_file][file]["pid"].append(pid)
                pid_ls = list(set(relationships[parent_file][file]["pid"]))
                relationships[parent_file][file]["pid"] = pid_ls
                relationships[parent_file][file]["parent_hostname"] = list(set(relationships[parent_file][file]["parent_hostname"] + parent_info['hostname']))
                relationships[parent_file][file]["user"] = list(set(parent_info['user'] + relationships[parent_file][file]["user"]))
        return relationships
    

def process_relations_csv(csv_file_path):
    """
    Extracts parent process IDs (PPIDs) and process IDs (PIDs) from Sysmon logs in CSV format.

    Args:
        csv_file_path: The path to the Sysmon CSV file.

    Returns:
        A dictionary indexed by nested keys of PPID and PID.
    """
    relationships = {}
    ppid_lookup = {}
    global orphan_cnt
    orphan_cnt = 0
    with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                event_id = int(row.get("EventID"))
                if event_id == 1:  # EventID 1: Process creation
                    relationships, ppid_lookup = process_creation_csv(row, relationships, ppid_lookup)
                elif event_id == 12 or event_id == 13:  # EventID 12 or 13: Registry operations
                    relationships = registry_operations_csv(row, relationships)
                elif event_id == 11 or event_id == 2:  # EventID 11: File creation or 2: File Process Change
                    relationships = file_creation_change_csv(row, relationships, ppid_lookup, event_id)
                elif event_id == 5:  # EventID 5: Process terminated
                    pid = int(row.get("pid"))
                    try:
                        for parent_file, parent_info in relationships.items():
                            if isinstance(parent_info, dict):
                                for file, child_node_details in parent_info.items():
                                    if isinstance(child_node_details, dict) and "pid" in child_node_details:
                                        pid = int(row.get("pid"))
                                        if  pid in  child_node_details.get("pid"):
                                            relationships[parent_file][file]["end_time"].append(row.get("UtcTime"))
                                            end_time = datetime.datetime.strptime(row.get("UtcTime"),"%m/%d/%y  %I:%M:%S %p")
                                            start_time = datetime.datetime.strptime(relationships[parent_file][file]["start_time"][0],"%m/%d/%y  %I:%M:%S %p")
                                            difference = end_time - start_time
                                            relationships[parent_file][file]["duration_min"].append(str(difference.seconds/60))
                    except (ValueError, TypeError) as e:
                        print(f"Error processing row entry: {e}\n{row}\n{relationships[parent_file][file]}")
                        print(traceback.format_exc())                       
            except (ValueError, TypeError) as e:
                print(f"\nError parsing numerical values in row error: {e}")
                print(traceback.format_exc())
                continue
            except (AttributeError, KeyError) as e:
                print(f"An unexpected error occurred: {e}")
                print(traceback.format_exc())
                continue
    print(f"Total orphan processes: {orphan_cnt}")
    return relationships
