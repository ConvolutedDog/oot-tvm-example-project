#!/bin/bash

# Script: gitfetch.sh
# Purpose: Automatically retry `git fetch` with arguments until success
#          or max retries reached.
# Handles timeouts (e.g., due to the FUCKING network provider) and other failures(e.g., the FUCKING DNS spoofing).
# Usage: ./gitfetch.sh [any git fetch arguments, e.g. upstream main]
# Note: It's user's pesponsibility for safe command.
#       Only works well on Linux systems. We have not test it on MacOS.

# Maximum number of retries
MAX_RETRIES=100
retry_count=0

# Timeout for git fetch (in seconds)
FETCH_TIMEOUT=20

# Loop until git fetch succeeds or the maximum number of retries is reached
while [ $retry_count -lt $MAX_RETRIES ]; do
    echo "Attempting git fetch ${@}... (Attempt: $((retry_count + 1))/$MAX_RETRIES)"

    # Execute git fetch with arguments and a timeout
    timeout $FETCH_TIMEOUT git fetch "$@"

    # Check the exit status of git fetch
    fetch_exit_code=$?
    if [ $fetch_exit_code -eq 0 ]; then
        echo "Git fetch succeeded!"
        exit 0  # Exit the script if successful
    elif [ $fetch_exit_code -eq 124 ]; then
        # Git fetch timed out (likely waiting for user input)
        echo "Git fetch timed out (possibly waiting for user input). Retrying..."
    else
        # Git fetch failed for other reasons
        echo "Git fetch failed with exit code $fetch_exit_code. Retrying..."
    fi

    retry_count=$((retry_count + 1))
    sleep 1  # Wait 1 seconds before retrying
done

# If the maximum number of retries is reached without success
echo "Maximum retries reached ($MAX_RETRIES attempts), git fetch still failed."
exit 1  # Exit the script with an error code
