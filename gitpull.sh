#!/bin/bash

# Script: gitpull.sh
# Purpose: Automatically retry `git pull` with arguments until success
#          or max retries reached.
# Handles timeouts (e.g., due to network provider) and other failures.
# Usage: ./gitpull.sh [any git pull arguments, e.g. upstream main]
# Note: It's user's pesponsibility for safe command.
#       Only works well on Linux systems. We have not test it on MacOS.

# Maximum number of retries
MAX_RETRIES=100
retry_count=0

# Timeout for git pull (in seconds)
PULL_TIMEOUT=20

# Loop until git pull succeeds or the maximum number of retries is reached
while [ $retry_count -lt $MAX_RETRIES ]; do
    echo "Attempting git pull ${@}... (Attempt: $((retry_count + 1))/$MAX_RETRIES)"

    # Execute git pull with arguments and a timeout
    timeout $PULL_TIMEOUT git pull "$@"

    # Check the exit status of git pull
    pull_exit_code=$?
    if [ $pull_exit_code -eq 0 ]; then
        echo "Git pull succeeded!"
        exit 0  # Exit the script if successful
    elif [ $pull_exit_code -eq 124 ]; then
        # Git pull timed out (likely waiting for user input)
        echo "Git pull timed out (possibly waiting for user input). Retrying..."
    else
        # Git pull failed for other reasons
        echo "Git pull failed with exit code $pull_exit_code. Retrying..."
    fi

    retry_count=$((retry_count + 1))
    sleep 1  # Wait 1 seconds before retrying
done

# If the maximum number of retries is reached without success
echo "Maximum retries reached ($MAX_RETRIES attempts), git pull still failed."
exit 1  # Exit the script with an error code
