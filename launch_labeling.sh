#!/usr/bin/env bash
# Description: Double-click launcher for Linux desktop environments.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

if [ ! -d "bubbly_flows" ]; then
    echo "ERROR: 'bubbly_flows' directory not found in: $SCRIPT_DIR"
    echo "Make sure this script is in the root of the Bubble-tracking repository."
    read -r -p "Press Enter to exit..."
    exit 1
fi

bash bubbly_flows/scripts/xanylabel.sh
