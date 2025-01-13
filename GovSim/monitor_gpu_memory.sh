#!/bin/bash
# File for monitoring GPU memory usage during command execution.
# Usage: ./monitor_gpu_memory.sh "<command>" <output_file>

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 \"<command>\" <output_file>"
    exit 1
fi

COMMAND=$1
OUTPUT_FILE=$2
TMP_LOG="gpu_memory.log"
MAX_MEMORY=0

cleanup() {
    rm -f $TMP_LOG
}
trap cleanup EXIT

module purge
module load 2023
module load Anaconda3/2023.07-2

source activate GovComGPTQ

nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 > $TMP_LOG &
NVIDIA_SMI_PID=$!

eval "$COMMAND" &> "$OUTPUT_FILE"

kill $NVIDIA_SMI_PID

if [ -f $TMP_LOG ]; then
    MAX_MEMORY=$(awk 'BEGIN {max=0} {if ($1 > max) max=$1} END {print max}' $TMP_LOG)
else
    echo "Error: GPU monitoring log file not found."
    exit 1
fi

echo "Maximum GPU Memory Used: ${MAX_MEMORY} MiB"
