import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from power_grid_model import initialize_array
from power_grid_model.utils import msgpack_serialize_to_file
from power_grid_model.validation import assert_valid_input_data
from power_grid_model_io.converters import VisionExcelConverter

vision_config = Path(__file__).parent / "vision_en.yaml"


def load_input(test_case, data_path):
    input_file = data_path / "vnf_excel" / f"{test_case}.xlsx"
    converter = VisionExcelConverter(source_file=input_file)
    converter.set_mapping_file(vision_config)
    input_data, extra_info = converter.load_input_data()
    # set large sk for source because Vision treats source as ideal voltage source
    input_data["source"]["sk"] = 1e20
    assert_valid_input_data(input_data)
    return input_data, extra_info


# profile
def load_profile(data_path, input_data, extra_info, n_steps):
    profile_file = data_path / "rail_profiles_OS LEEUWARDEN_2022.csv"
    profile_df = pd.read_csv(profile_file, skiprows=[1], nrows=n_steps, engine="c")
    del profile_df["*.NewPV_*"]
    n_steps = profile_df.shape[0]
    profile_df["Date & Time"] = pd.to_datetime(profile_df["Date & Time"], format="%d-%m-%Y %H:%M")
    profile_df = profile_df.set_index("Date & Time")
    profile_df *= 1e3
    col_node_ids = [x.strip(".*") for x in profile_df.columns]
    node_ids_index = pd.Index(col_node_ids)
    all_load_node_ids = []
    for pgm_node_id in input_data["sym_load"]["id"]:
        info = extra_info[pgm_node_id]
        if "Node.ID" in info:
            node_id = info["Node.ID"]
            if isinstance(node_id, float):
                all_load_node_ids.append(str(int(node_id)))
            else:
                all_load_node_ids.append(str(node_id))
        else:
            all_load_node_ids.append("")
    indexer = node_ids_index.get_indexer(all_load_node_ids)
    has_profile = indexer >= 0
    loads_with_profile = input_data["sym_load"][has_profile]
    indexer = indexer[has_profile]
    load_update = initialize_array("update", "sym_load", (n_steps, indexer.size))
    load_update["id"] = loads_with_profile["id"].reshape(1, -1)
    load_update["p_specified"] = profile_df.to_numpy()[:, indexer]
    non_zero_index = loads_with_profile["p_specified"] != 0.0
    load_update["q_specified"][:, non_zero_index] = (
        loads_with_profile["q_specified"][non_zero_index] / loads_with_profile["p_specified"][non_zero_index]
    ).reshape(1, -1) * load_update["p_specified"][:, non_zero_index]
    update_data = {"sym_load": load_update}
    return update_data, n_steps


def get_id_map(extra_info):
    name_map = {
        "Nodes": "Nodes",
        "Cables": "kabel",
        "Links": "link",
        "Transformers": "transformator",
        "Loads": "belasting",
        "Earthing transformers": "nulpuntstransformator",
        "Sources": "netvoeding",
    }
    id_map = {}
    for i, info in extra_info.items():
        ref = info["id_reference"]
        name = name_map[ref["table"]]
        if name == "Nodes":
            number = ref["key"]["Number"]
            value = f"{number}"
        elif "Subnumber" in ref["key"]:
            sub_number = ref["key"]["Subnumber"]
            node_number = ref["key"]["Node.Number"]
            value = f"{name}.{node_number}.{sub_number}"
        else:
            number = ref["key"]["Number"]
            value = f"{name}.{number}"
        id_map[i] = value
    return pd.Series(id_map)


def load_single_result_type(df, input_data, pgm_component, id_map, attribute_map):
    dfs = []
    dfs_matching_ids = []
    dfs_matching_col_idx = []
    input_data = input_data[pgm_component]
    ids = input_data["id"]
    col_names = id_map.loc[ids]

    for attr in attribute_map.values():
        attr_df = df.loc[:, (slice(None), attr)].droplevel(1, axis=1)
        dfs.append(attr_df)
        col_idx = attr_df.columns.get_indexer(col_names)
        id_found = col_idx >= 0
        matching_ids = ids[id_found]
        matching_col_idx = col_idx[id_found]
        dfs_matching_ids.append(matching_ids)
        dfs_matching_col_idx.append(matching_col_idx)

    if len(attribute_map) > 1:
        for ids1, ids2 in zip(dfs_matching_ids[:1], dfs_matching_ids[1:]):
            assert np.all(ids1 == ids2)
    matching_ids = dfs_matching_ids[0]
    matching_col_idx = dfs_matching_col_idx[0]

    result_array = initialize_array("sym_output", pgm_component, shape=(df.shape[0], matching_ids.size))
    result_array["id"] = matching_ids.reshape(1, -1)
    for pgm_attr, attr_df in zip(attribute_map.keys(), dfs):
        result_array[pgm_attr] = attr_df.iloc[:, matching_col_idx]

    return result_array


def load_single_result_sheet(
    result_path, input_data, id_map, n_steps, sheet, pgm_components, attribute_map, multiplier
):
    df = pd.read_csv(result_path / f"{sheet}.csv", skiprows=[2], nrows=n_steps, engine="c", header=[0, 1]).iloc[:, 1:]
    df *= multiplier
    result_dict = {}
    for pgm_component in pgm_components:
        result_dict[pgm_component] = load_single_result_type(df, input_data, pgm_component, id_map, attribute_map)

    return result_dict


def load_vision_results(test_case, data_path, input_data, extra_info, n_steps):
    result_path = data_path / "csv_result" / test_case
    id_map = get_id_map(extra_info)

    node_output = load_single_result_sheet(
        result_path,
        input_data,
        id_map,
        n_steps,
        sheet="node",
        pgm_components=["node"],
        attribute_map={"u": "U"},
        multiplier=1e3,
    )
    branch_output = load_single_result_sheet(
        result_path,
        input_data,
        id_map,
        n_steps,
        sheet="branch",
        pgm_components=["line", "transformer", "link"],
        attribute_map={"p_from": "P1", "p_to": "P2", "q_from": "Q1", "q_to": "Q2"},
        multiplier=1e6,
    )
    appliance_output = load_single_result_sheet(
        result_path,
        input_data,
        id_map,
        n_steps,
        sheet="appliance",
        pgm_components=["sym_load", "source"],
        attribute_map={"p": "P", "q": "Q"},
        multiplier=1e6,
    )

    return {**node_output, **branch_output, **appliance_output}


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def dump_to_file(data_path: Path, test_case, input_data, update_data, result_data, extra_info):
    output_path: Path = data_path / "PGM_data" / test_case
    output_path.mkdir(parents=True, exist_ok=True)
    msgpack_serialize_to_file(file_path=output_path / "input.pgmb", data=input_data, use_compact_list=True)
    msgpack_serialize_to_file(file_path=output_path / "update.pgmb", data=update_data, use_compact_list=True)
    msgpack_serialize_to_file(file_path=output_path / "result.pgmb", data=result_data, use_compact_list=True)

    with open(output_path / "extra_info.json", "w") as f:
        json.dump(extra_info, f, cls=NpEncoder, indent=2)


# noinspection DuplicatedCode
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=Path, help="Path to the data")
    parser.add_argument("--n-steps", type=int, help="Number of steps to run")
    parser.add_argument("--test-case", type=str, help="Name of test case", default="Leeuwarden_small_P")
    args = parser.parse_args()
    data_path = args.data_path
    n_steps = args.n_steps
    test_case = args.test_case

    input_data, extra_info = load_input(test_case, data_path)
    update_data, n_steps = load_profile(data_path, input_data, extra_info, n_steps)
    result_data = load_vision_results(test_case, data_path, input_data, extra_info, n_steps)

    dump_to_file(data_path, test_case, input_data, update_data, result_data, extra_info)


if __name__ == "__main__":
    main()
