#!/bin/bash

data="$1"
pipeline_file="$2"
output_directory="$3"

slice_count="$4"
slices_per_chunk="$5"
chunk_count=$(((slice_count + slices_per_chunk - 1) / slices_per_chunk))

for ((i=0; i<chunk_count; i++)); do
	start=$((i * slices_per_chunk))
	stop=$((start + slices_per_chunk))
	stop=$((stop < slice_count ? stop : slice_count))

	python ./pipeline_preprocessor.py -i $pipeline_file -o $pipeline_file --slice-start $start --slice-stop $stop

	httomo run $data $pipeline_file $output_directory
done
