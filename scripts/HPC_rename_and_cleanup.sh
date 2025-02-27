#!/bin/bash

# Define paths
results_dir="../results/"
ologs_dir="./ologs/"
elogs_dir="./elogs/"

# Set the name counter
highest_num=$(ls "$results_dir" | grep -oP 'test\K\d+' | sort -n | tail -1)
counter=$((highest_num + 1))

# Rename new results
for folder in "$results_dir"/*.pbs
do
  new_folder_name="test$counter"
  mv "$folder" "$results_dir/$new_folder_name"
  ((counter++))
done

# Clear ologs and elogs
rm -f "$ologs_dir"/*.pbs.OU
rm -f "$elogs_dir"/*.pbs.ER