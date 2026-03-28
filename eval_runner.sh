#!/usr/bin/env bash
set -uo pipefail

# Experiment configs
N_SHOTS=0
OUTPUT_DIR="outputs"
# Change this to run on the entire dataset instead of the sample
DATASET_PATH="data/sake-dataset-sample.json"
# End config

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$SCRIPT_DIR/scripts"

if [[ ! -d "$SCRIPTS_DIR" ]]; then
	echo "Error: scripts directory not found at $SCRIPTS_DIR" >&2
	exit 1
fi

mapfile -t script_files < <(find "$SCRIPTS_DIR" -maxdepth 1 -type f -name "*.sh" | sort)

if [[ ${#script_files[@]} -eq 0 ]]; then
	echo "No .sh files found in $SCRIPTS_DIR"
	exit 0
fi

for script in "${script_files[@]}"; do
	echo "Running: $script"
	bash "$script" --dataset_path "${DATASET_PATH}" --n_shots "${N_SHOTS}" --output_dir "${OUTPUT_DIR}"
done

echo "Completed running ${#script_files[@]} script(s)."
