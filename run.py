from pathlib import Path

import pandas as pd
import numpy as np
from power_grid_model import PowerGridModel
from power_grid_model.utils import msgpack_deserialize_from_file
import argparse


def run(input_data, update_data, calculation_method):
    pgm = PowerGridModel(input_data)
    pgm_result = pgm.calculate_power_flow(update_data=update_data, calculation_method=calculation_method)
    print("----Node Result---")
    print(pd.DataFrame(pgm_result["node"][0, :]).head())
    return pgm, pgm_result


def load_data(data_path, test_case, n_steps):
    path = data_path / "PGM_data" / test_case
    input_data = msgpack_deserialize_from_file(path / "input.pgmb")
    update_data = msgpack_deserialize_from_file(path / "update.pgmb")

    if n_steps is not None:
        update_data = {k: v[:n_steps, ...] for k, v in update_data.items()}

    return input_data, update_data


# noinspection DuplicatedCode
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=Path, help="Path to the data")
    parser.add_argument("--n-steps", type=int, help="Number of steps to run")
    parser.add_argument("--test-case", type=str, help="Name of test case")
    parser.add_argument("--calculation_method", type=str, help="Calculation method", default="newton_raphson")

    args = parser.parse_args()
    data_path = args.data_path
    n_steps = args.n_steps
    test_case = args.test_case
    calculation_method = args.calculation_method

    input_data, update_data = load_data(data_path, test_case, n_steps)

    run(input_data, update_data, calculation_method)


if __name__ == "__main__":
    main()
