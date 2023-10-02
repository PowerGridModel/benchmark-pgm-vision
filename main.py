import os
from pathlib import Path

import pandas as pd
import numpy as np
from power_grid_model import PowerGridModel, initialize_array
from power_grid_model.validation import assert_valid_input_data
from power_grid_model_io.converters import VisionExcelConverter
import argparse

vision_config = Path(__file__).parent / "vision_en.yaml"


# input
def load_input(test_case, data_path):
    input_file = data_path / "excel_input" / f"{test_case}.xlsx"
    converter = VisionExcelConverter(source_file=input_file)
    converter.set_mapping_file(vision_config)
    input_data, extra_info = converter.load_input_data()
    # set large sk for source because Vision treats source as ideal voltage source
    input_data["source"]["sk"] = 1e12
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
    all_load_node_ids = [str(extra_info[x].get("Node.ID", 0)) for x in input_data["sym_load"]["id"]]
    indexer = node_ids_index.get_indexer(all_load_node_ids)
    has_profile = indexer >= 0
    loads_with_profile = input_data["sym_load"][has_profile]
    indexer = indexer[has_profile]
    load_update = initialize_array("update", "sym_load", (n_steps, indexer.size))
    load_update["id"] = loads_with_profile["id"].reshape(1, -1)
    load_update["p_specified"] = profile_df.to_numpy()[:, indexer]
    load_update["q_specified"] = (loads_with_profile["q_specified"] / loads_with_profile["p_specified"]).reshape(
        1, -1
    ) * load_update["p_specified"]
    update_data = {"sym_load": load_update}
    return update_data


def get_node_name_mapping(input_data, extra_info):
    node_name_mapping = {}
    for pgm_id in input_data["node"]["id"]:
        info = extra_info[pgm_id]
        if "Name" not in info:
            continue
        name = info["Name"]
        if name not in node_name_mapping:
            node_name_mapping[name] = pgm_id
        else:
            number = 1
            while f"{name}.{number}" in node_name_mapping:
                number += 1
            node_name_mapping[f"{name}.{number}"] = pgm_id
    return node_name_mapping


def compare_result(data_path, input_data, pgm, pgm_result, extra_info, test_case, n_steps):
    vision_result_file = data_path / "excel_result" / f"{test_case}.node.csv"
    vision_df = pd.read_csv(vision_result_file, skiprows=[1, 2], nrows=n_steps, engine="c").set_index("Naam")
    vision_df.index.name = "Date & Time"
    vision_df *= 1e3
    node_names = vision_df.columns.to_list()
    node_name_mapping = get_node_name_mapping(input_data, extra_info)
    vision_node_indexer = []
    node_pgm_ids = []
    for i, name in enumerate(node_names):
        if name in node_name_mapping:
            vision_node_indexer.append(i)
            node_pgm_ids.append(node_name_mapping[name])
    vision_node_indexer = np.array(vision_node_indexer, dtype=np.int64)
    node_pgm_ids = np.array(node_pgm_ids, dtype=np.int32)
    pgm_node_indexer = pgm.get_indexer("node", node_pgm_ids)
    u_pgm = pgm_result["node"][:, pgm_node_indexer]["u"]
    u_vision = vision_df.iloc[:, vision_node_indexer].to_numpy()
    diff = np.abs(u_vision - u_pgm)
    max_diff_per_node = np.max(diff, axis=0)
    max_diff_pu_per_node = np.max(diff / input_data["node"][pgm_node_indexer]["u_rated"].reshape(1, -1), axis=0)
    max_diff = np.max(max_diff_per_node)
    max_diff_pu = np.max(max_diff_pu_per_node)
    print(f"Max voltage deviation: {max_diff} V.")
    print(f"Max voltage deviation: {max_diff_pu} pu.")
    mv_select = input_data["node"][pgm_node_indexer]["u_rated"] > 1e3
    max_diff_mv = np.max(max_diff_per_node[mv_select])
    max_diff_pu_mv = np.max(max_diff_pu_per_node[mv_select])
    max_diff_lv = np.max(max_diff_per_node[~mv_select])
    max_diff_pu_lv = np.max(max_diff_pu_per_node[~mv_select])
    print(f"Max MV voltage deviation: {max_diff_mv} V.")
    print(f"Max MV voltage deviation: {max_diff_pu_mv} pu.")
    print(f"Max LV voltage deviation: {max_diff_lv} V.")
    print(f"Max LV voltage deviation: {max_diff_pu_lv} pu.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=Path, help="Path to the data")
    parser.add_argument("--n-steps", type=int, help="Number of steps to run")
    parser.add_argument("--test-case", type=str, help="Name of test case")
    args = parser.parse_args()
    data_path = args.data_path
    n_steps = args.n_steps
    test_case = args.test_case

    input_data, extra_info = load_input(test_case, data_path)
    update_data = load_profile(data_path, input_data, extra_info, n_steps)

    pgm = PowerGridModel(input_data)
    pgm_result = pgm.calculate_power_flow(update_data=update_data)
    print(pd.DataFrame(pgm_result["node"][0, :]))

    compare_result(data_path, input_data, pgm, pgm_result, extra_info, test_case, n_steps)


if __name__ == "__main__":
    main()
