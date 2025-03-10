#!/bin/bash

# Maximum number of retries
MAX_RETRIES=100
retry_count=0

# Timeout for git push (in seconds)
PUSH_TIMEOUT=20

# Loop until git push succeeds or the maximum number of retries is reached
while [ $retry_count -lt $MAX_RETRIES ]; do
    echo "Attempting git push... (Attempt: $((retry_count + 1))/$MAX_RETRIES)"

    # Execute git push with a timeout
    timeout $PUSH_TIMEOUT git push

    # Check the exit status of git push
    push_exit_code=$?
    if [ $push_exit_code -eq 0 ]; then
        echo "Git push succeeded!"
        exit 0  # Exit the script if successful
    elif [ $push_exit_code -eq 124 ]; then
        # Git push timed out (likely waiting for user input)
        echo "Git push timed out (possibly waiting for user input). Retrying..."
    else
        # Git push failed for other reasons
        echo "Git push failed with exit code $push_exit_code. Retrying..."
    fi

    retry_count=$((retry_count + 1))
    sleep 1  # Wait 1 seconds before retrying
done

# If the maximum number of retries is reached without success
echo "Maximum retries reached ($MAX_RETRIES attempts), git push still failed."
exit 1  # Exit the script with an error code
