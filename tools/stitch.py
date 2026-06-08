import argparse
import h5py
import os
from pathlib import Path
from datetime import datetime


def httomo_output_sort_key(dir_name):
    date_str = dir_name.replace("_output", "")
    return datetime.strptime(date_str, "%d-%m-%Y_%H_%M_%S")


def main(args):
    directories = [
        d
        for d in os.listdir(args.i)
        if os.path.isdir(os.path.join(args.i, d)) and d.endswith("_output")
    ]
    h5_file_paths = []

    directories.sort(key=httomo_output_sort_key)
    for dir in directories:
        dir_path = os.path.abspath(os.path.join(args.i, dir))
        h5_files = [f for f in os.listdir(dir_path) if f.endswith(".h5")]
        h5_file_paths.extend([os.path.join(dir_path, f) for f in h5_files])

    with h5py.File(h5_file_paths[0], "r") as f:
        dataset = f["data"]
        shape = dataset.shape
        dtype = dataset.dtype

    with h5py.File(h5_file_paths[-1], "r") as f:
        dataset = f["data"]
        last_shape = dataset.shape

    chunk_count = len(h5_file_paths)
    slicing_dim = 1
    global_shape = list(shape)
    global_shape[slicing_dim] *= chunk_count - 1
    global_shape[slicing_dim] += list(last_shape)[slicing_dim]

    output_file_path = os.path.join(args.i, "stitched_output.h5")
    with h5py.File(output_file_path, "w") as output:
        dataset = output.create_dataset("data", global_shape, dtype)

        for i, x in enumerate(h5_file_paths):
            with h5py.File(x, "r") as f:
                dataset_chunk = f["data"]
                slice_count = dataset_chunk.shape[slicing_dim]
                start_slice = i * slice_count
                dataset[:, start_slice : start_slice + slice_count, :] = dataset_chunk


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=Path, required=True)
    args = parser.parse_args()

    main(args)
