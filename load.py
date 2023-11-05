from pathlib import Path

from power_grid_model.utils import msgpack_deserialize_from_file

from time import time


path = Path(__file__).parent / "data" / "PGM_data" / "Leeuwarden_big_P" / "result.pgmb"

file_size = path.stat().st_size / 1e9

start = time()
data = msgpack_deserialize_from_file(path)
end = time()
print(f"Time to load: {end - start} seconds.")

buffer_size = 0
for v in data.values():
    buffer_size += v.itemsize * v.size / 1e9

print(f"Memory usage of file size: {file_size} GB.")
print(f"Memory usage of buffers: {buffer_size} GB.")
print(f"Total unavoidable memory usage: {file_size + buffer_size} GB.")
