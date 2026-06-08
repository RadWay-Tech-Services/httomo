import argparse
import yaml
from pathlib import Path


def main(args):
    with open(args.i, "r") as input_yaml:
        pipeline = yaml.safe_load(input_yaml)

    slice_preview = pipeline[0]["parameters"]["preview"]["detector_y"]
    slice_preview["start"] = args.slice_start
    slice_preview["stop"] = args.slice_stop

    with open(args.o, "w") as output_yaml:
        yaml.dump(pipeline, output_yaml, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=Path, required=True)
    parser.add_argument("-o", type=Path, required=True)
    parser.add_argument("--slice-start", type=int, required=True)
    parser.add_argument("--slice-stop", type=int, required=True)
    args = parser.parse_args()

    main(args)
