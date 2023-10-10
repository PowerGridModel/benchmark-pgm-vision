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
    path = data_path / "PGM_data" / test_case
    input_data = msgpack_deserialize_from_file(path / "input.pgmb")
    update_data = msgpack_deserialize_from_file(path / "update.pgmb")
    vision_result = msgpack_deserialize_from_file(path / "result.pgmb")

    if n_steps is not None:
        update_data = {k: v[:n_steps, ...] for k, v in update_data.items()}
    else:
        n_steps = next(iter(update_data.values())).shape[0]

    return input_data, update_data, vision_result, n_steps


def compare_results(pgm, pgm_result, vision_result, input_data):
    compare_nodes(pgm, pgm_result["node"], vision_result["node"], input_data["node"])


def compare_nodes(pgm, pgm_result, vision_result, input_data):
    node_indexer = pgm.get_indexer("node", vision_result[0, :]["id"])
    pgm_result = pgm_result[:, node_indexer]
    input_data = input_data[node_indexer]
    diff = np.abs(pgm_result["u"] - vision_result["u"])
    max_diff_per_node = np.max(diff, axis=0)
    max_diff_pu_per_node = np.max(diff / input_data["u_rated"].reshape(1, -1), axis=0)
    max_diff = np.max(max_diff_per_node)
    max_diff_pu = np.max(max_diff_pu_per_node)
    hvmv_select = input_data["u_rated"] > 1e3
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
    compare_results(pgm, pgm_result, vision_result, input_data)


if __name__ == "__main__":
    main()
