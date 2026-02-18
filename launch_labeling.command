#!/bin/bash
# Description: Double-click launcher for Bubbly Flows X-AnyLabeling on macOS/Linux.

# Get the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Check if bubbly_flows exists
if [ ! -d "bubbly_flows" ]; then
    echo "ERROR: 'bubbly_flows' directory not found in: $DIR"
    echo "Make sure this script is in the root of the Bubbly-tracking repository."
    read -p "Press Enter to exit..."
    exit 1
fi

# Run the labeler script
# We use 'bash' explicitly in case the file doesn't have +x bits set somehow
source bubbly_flows/scripts/xanylabel.sh
