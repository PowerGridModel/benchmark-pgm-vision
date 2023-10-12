from pathlib import Path

import pandas as pd
import numpy as np
from power_grid_model import PowerGridModel
from power_grid_model.utils import msgpack_deserialize_from_file
import argparse
from time import time


def run(input_data, update_data, calculation_method, n_steps):
    start = time()
    pgm = PowerGridModel(input_data)
    end = time()
    print(f"\n---Model initialization---")
    print(f"Time cost: {end - start} seconds")
    print("Components count")
    print(pgm.all_component_count)
    start = time()
    pgm_result = pgm.calculate_power_flow(update_data=update_data, calculation_method=calculation_method)
    end = time()
    print(f"\n---Calculation---")
    print(f"Method: {calculation_method}")
    print(f"Steps: {n_steps}")
    print(f"Time cost: {end - start} seconds")
    print("\n\n---Node Result---")
    print(pd.DataFrame(pgm_result["node"][0, :]).head())
    return pgm, pgm_result


def load_data(data_path, test_case, n_steps):
    print("\n\n---Load data---")
    start = time()
    path = data_path / "PGM_data" / test_case
    vision_result = msgpack_deserialize_from_file(path / "result.pgmb")
    input_data = msgpack_deserialize_from_file(path / "input.pgmb")
    update_data = msgpack_deserialize_from_file(path / "update.pgmb")

    if n_steps is not None:
        update_data = {k: v[:n_steps, ...] for k, v in update_data.items()}
        vision_result = {k: v[:n_steps, ...] for k, v in vision_result.items()}
    else:
        n_steps = next(iter(update_data.values())).shape[0]
    simplify_result_data(vision_result)
    end = time()
    print(f"Time cost: {end - start} seconds")

    return input_data, update_data, vision_result, n_steps


def simplify_result_data(result_data):
    # modify result data in place
    for name, array in result_data.items():
        ids = array[0, :]["id"]
        if name == "node":
            new_array = {"id": ids, "u": array["u"]}
        elif name in {"sym_load", "source", "shunt"}:
            new_array = {"id": ids, "s": array["p"] + 1j * array["q"]}
        elif name in {"link", "line", "transformer"}:
            new_array = {
                "id": ids,
                "s": np.stack((array["p_from"] + 1j * array["q_from"], array["p_to"] + 1j * array["q_to"]), axis=2),
            }
        else:
            new_array = array
        result_data[name] = new_array


def compare_results(pgm_result, vision_result, input_data):
    print("\n\n---Total source power---")
    max_s = np.max(np.abs(pgm_result["source"]["s"]))
    print(f"Max apparent power of all sources: {max_s * 1e-6} MVA")
    compare_nodes(pgm_result["node"], vision_result["node"], input_data["node"])
    compare_branches(pgm_result, vision_result, "line")
    compare_branches(pgm_result, vision_result, "transformer")
    compare_branches(pgm_result, vision_result, "link")
    compare_appliances(pgm_result, vision_result, "sym_load")
    compare_appliances(pgm_result, vision_result, "source")


def index_by_vision(pgm, pgm_result, vision_result, input_data):
    # slide pgm input and result in place
    for name, vision_array in vision_result.items():
        indexer = pgm.get_indexer(name, vision_array["id"])
        pgm_result[name] = pgm_result[name][:, indexer]
        input_data[name] = input_data[name][indexer]


def compare_nodes(pgm_node_result, vision_node_result, node_input_data):
    diff = np.abs(pgm_node_result["u"] - vision_node_result["u"])
    max_diff_per_node = np.max(diff, axis=0)
    max_diff_pu_per_node = np.max(diff / node_input_data["u_rated"].reshape(1, -1), axis=0)
    max_diff = np.max(max_diff_per_node)
    max_diff_pu = np.max(max_diff_pu_per_node)
    hvmv_select = node_input_data["u_rated"] > 1e3
    max_diff_hvmv = np.max(max_diff_per_node[hvmv_select])
    max_diff_pu_hvmv = np.max(max_diff_pu_per_node[hvmv_select])
    max_diff_lv = np.max(max_diff_per_node[~hvmv_select])
    max_diff_pu_lv = np.max(max_diff_pu_per_node[~hvmv_select])
    print("\n\n---Node Comparison---")
    print(f"Max voltage deviation: {max_diff} V.")
    print(f"Max voltage deviation: {max_diff_pu} pu.")
    print(f"Max HV/MV voltage deviation: {max_diff_hvmv} V.")
    print(f"Max HV/MV voltage deviation: {max_diff_pu_hvmv} pu.")
    print(f"Max LV voltage deviation: {max_diff_lv} V.")
    print(f"Max LV voltage deviation: {max_diff_pu_lv} pu.")


def compare_branches(pgm_result, vision_result, component):
    vision_branch_result = vision_result[component]
    pgm_branch_result = pgm_result[component]
    diff = np.abs(vision_branch_result["s"] - pgm_branch_result["s"])
    diff_per_branch = np.max(diff, axis=(0, 2))
    max_diff = np.max(diff_per_branch)
    print(f"\n\n---{component} Comparison---")
    print(f"Max complex power deviation: {max_diff * 1e-6} MVA")


def compare_appliances(pgm_result, vision_result, component):
    vision_app_result = vision_result[component]
    pgm_app_result = pgm_result[component]
    diff = np.abs(vision_app_result["s"] - pgm_app_result["s"])
    diff_per_app = np.max(diff, axis=0)
    max_diff = np.max(diff_per_app)
    print(f"\n\n---{component} Comparison---")
    print(f"Max complex power deviation: {max_diff * 1e-6} MVA")


# noinspection DuplicatedCode
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=Path, help="Path to the data")
    parser.add_argument("--n-steps", type=int, help="Number of steps to run")
    parser.add_argument("--test-case", type=str, help="Name of test case")
    parser.add_argument("--calculation-method", type=str, help="Calculation method", default="newton_raphson")

    args = parser.parse_args()
    data_path = args.data_path
    n_steps = args.n_steps
    test_case = args.test_case
    calculation_method = args.calculation_method

    print(f"Test case: {test_case}")

    input_data, update_data, vision_result, n_steps = load_data(data_path, test_case, n_steps)
    pgm, pgm_result = run(input_data, update_data, calculation_method, n_steps)
    del update_data
    index_by_vision(pgm, pgm_result, vision_result, input_data)
    simplify_result_data(pgm_result)
    compare_results(pgm_result, vision_result, input_data)


if __name__ == "__main__":
    main()
