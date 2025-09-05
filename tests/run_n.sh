#!/bin/bash

# Usage: ./runN.sh <count> <command> [args...]

# Check arguments
if [ $# -lt 2 ]; then
  echo "Usage: $0 <count> <command> [args...]"
  exit 1
fi

count=$1
shift   # remove the first argument so "$@" is the command + args

# Run the command 'count' times in parallel
for i in $(seq 1 $count); do
  "$@" &
done

# Wait for all background processes
wait
