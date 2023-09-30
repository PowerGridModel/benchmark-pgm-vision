import os
from pathlib import Path

import pandas as pd
import numpy as np
from power_grid_model import PowerGridModel, initialize_array
from power_grid_model.validation import assert_valid_input_data
from power_grid_model_io.converters import VisionExcelConverter

vision_config = Path(__file__).parent / "vision_en.yaml"
DATA_PATH = Path(os.environ["PGM_VISION_DATA_PATH"])


# input
def load_input(test_case):
    input_file = DATA_PATH / "excel_input" / f"{test_case}.xlsx"
    converter = VisionExcelConverter(source_file=input_file)
    converter.set_mapping_file(vision_config)
    input_data, extra_info = converter.load_input_data()
    # set large sk for source because Vision treats source as ideal voltage source
    input_data["source"]["sk"] = 1e12
    assert_valid_input_data(input_data)
    return input_data, extra_info


# profile
def load_profile(input_data, extra_info, n_steps):
    profile_file = DATA_PATH / "rail_profiles_OS LEEUWARDEN_2022.xlsx"
    profile_df = pd.read_excel(profile_file, sheet_name=0, skiprows=[1], nrows=n_steps)
    del profile_df["*.NewPV_*"]
    profile_df["Date & Time"] = pd.to_datetime(
        profile_df["Date & Time"], format="%d-%m-%Y %H:%M"
    )
    profile_df = profile_df.set_index("Date & Time")
    profile_df *= 1e3
    col_node_ids = [x.strip(".*") for x in profile_df.columns]
    node_ids_index = pd.Index(col_node_ids)
    all_load_node_ids = [
        str(int(extra_info[x].get("Node.ID", 0))) for x in input_data["sym_load"]["id"]
    ]
    indexer = node_ids_index.get_indexer(all_load_node_ids)
    has_profile = indexer >= 0
    loads_with_profile = input_data["sym_load"][has_profile]
    indexer = indexer[has_profile]
    load_update = initialize_array("update", "sym_load", (n_steps, indexer.size))
    load_update["id"] = loads_with_profile["id"].reshape(1, -1)
    load_update["p_specified"] = profile_df.to_numpy()[:, indexer]
    load_update["q_specified"] = (
        loads_with_profile["q_specified"] / loads_with_profile["p_specified"]
    ).reshape(1, -1) * load_update["p_specified"]
    update_data = {"sym_load": load_update}
    return update_data


def get_node_name_mapping(input_data, extra_info):
    node_name_mapping = {}
    for pgm_id, u_rated in zip(input_data["node"]["id"], input_data["node"]["u_rated"]):
        info = extra_info[pgm_id]
        if "Name" not in info:
            continue
        name = info["Name"]
        if name not in node_name_mapping:
            node_name_mapping[name] = []
        node_name_mapping[name].append((pgm_id, u_rated))
    node_name_mapping = {
        k: v for k, v in node_name_mapping.items() if check_u_rated_no_duplicates(v)
    }
    return node_name_mapping


def check_u_rated_no_duplicates(info):
    u_rated = np.array([x[1] for x in info])
    return u_rated.size == np.unique(u_rated).size


def get_closest_node(nodes, avg_u):
    dev = np.inf
    found_id = None
    for pgm_id, u_rated in nodes:
        if np.abs(avg_u - u_rated) < dev:
            found_id = pgm_id
            dev = np.abs(avg_u - u_rated)
    return found_id


def compare_result(input_data, pgm, pgm_result, extra_info, test_case, n_steps):
    vision_result_file = DATA_PATH / "excel_result" / f"{test_case}.xlsx"
    vision_df = pd.read_excel(
        vision_result_file, sheet_name="Knooppunten", skiprows=[1, 2], nrows=n_steps
    ).set_index("Naam")
    vision_df.index.name = "Date & Time"
    vision_df *= 1e3
    node_names = vision_df.columns.to_list()
    node_name_mapping = get_node_name_mapping(input_data, extra_info)
    vision_node_indexer = []
    node_pgm_ids = []
    for i, name in enumerate(node_names):
        if "." in name:
            name = name[:-2]
        if name in node_name_mapping:
            vision_node_indexer.append(i)
            node_pgm_ids.append(
                get_closest_node(
                    node_name_mapping[name], np.mean(vision_df.iloc[:, i].to_numpy())
                )
            )
    vision_node_indexer = np.array(vision_node_indexer, dtype=np.int64)
    node_pgm_ids = np.array(node_pgm_ids, dtype=np.int32)
    pgm_node_indexer = pgm.get_indexer("node", node_pgm_ids)
    u_pgm = pgm_result["node"][:, pgm_node_indexer]["u"]
    u_vision = vision_df.iloc[:, vision_node_indexer].to_numpy()
    diff = np.abs(u_vision - u_pgm)
    max_diff = np.max(diff)
    max_diff_pu = np.max(
        diff / input_data["node"][pgm_node_indexer]["u_rated"].reshape(1, -1)
    )
    max_diff_per_node = np.max(diff, axis=0)
    print(f"Max voltage deviation: {max_diff} V.")
    print(f"Max voltage deviation: {max_diff_pu} pu.")
    node_names_large_dev = [
        node_names[vision_node_indexer[i]]
        for i, dev in enumerate(max_diff_per_node)
        if dev > 1.0
    ]
    print(f"Nodes with deviation larger than 1 volt: {len(node_names_large_dev)}")
    print(node_names_large_dev)


def main():
    test_case = "Leeuwarden_small_P"
    n_steps = 20

    input_data, extra_info = load_input(test_case)
    update_data = load_profile(input_data, extra_info, n_steps)

    pgm = PowerGridModel(input_data)
    pgm_result = pgm.calculate_power_flow(update_data=update_data)
    print(pd.DataFrame(pgm_result["node"][0, :]))

    compare_result(input_data, pgm, pgm_result, extra_info, test_case, n_steps)


if __name__ == "__main__":
    main()
