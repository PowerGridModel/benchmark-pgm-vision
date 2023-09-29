import os
from pathlib import Path

import pandas as pd
from power_grid_model import PowerGridModel, initialize_array
from power_grid_model.validation import assert_valid_input_data
from power_grid_model_io.converters import VisionExcelConverter

vision_config = Path(__file__).parent / "vision_en.yaml"
DATA_PATH = Path(os.environ["PGM_VISION_DATA_PATH"])


# input
def load_input(input_file):
    converter = VisionExcelConverter(source_file=input_file)
    converter.set_mapping_file(vision_config)
    input_data, extra_info = converter.load_input_data()
    # set large sk for source because Vision treats source as ideal voltage source
    input_data["source"]["sk"] = 1e12
    assert_valid_input_data(input_data)
    return input_data, extra_info


# profile
def load_profile(input_data, extra_info, n_steps, profile_file):
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


def main():
    test_case = "Leeuwarden_small_P"
    n_steps = 10
    input_file = DATA_PATH / "excel_input" / f"{test_case}.xlsx"
    profile_file = DATA_PATH / "rail_profiles_OS LEEUWARDEN_2022.xlsx"

    input_data, extra_info = load_input(input_file)
    update_data = load_profile(input_data, extra_info, n_steps, profile_file)

    pgm = PowerGridModel(input_data)
    pgm_result = pgm.calculate_power_flow(update_data=update_data)
    print(pd.DataFrame(pgm_result["node"][0, :]))


if __name__ == "__main__":
    main()
