from power_grid_model_io.converters import VisionExcelConverter
from power_grid_model import PowerGridModel
from power_grid_model.validation import assert_valid_input_data
import os
from pathlib import Path
import pandas as pd


vision_config = Path(__file__).parent / "vision_en.yaml"

DATA_PATH = Path(os.environ["PGM_VISION_DATA_PATH"])

test_case = "Leeuwarden_small_P"
input_file = DATA_PATH / "excel_input" / f"{test_case}.xlsx"

converter = VisionExcelConverter(source_file=input_file)
converter.set_mapping_file(vision_config)
input_data, extra_info = converter.load_input_data()
assert_valid_input_data(input_data)

# set large sk for source
input_data["source"]["sk"] = 1e12
pgm = PowerGridModel(input_data)
pgm_result = pgm.calculate_power_flow()
print(pd.DataFrame(pgm_result["node"]))
