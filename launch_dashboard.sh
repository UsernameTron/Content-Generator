#!/bin/bash
# Dashboard launcher script for healthcare contradiction detection system

# Default mode is regular
MODE=${1:-regular}

if [ "$MODE" != "regular" ] && [ "$MODE" != "testing" ]; then
    echo "Invalid mode: $MODE"
    echo "Usage: $0 [regular|testing]"
    exit 1
fi

# Set environment variable and launch dashboard
export DASHBOARD_MODE=$MODE
python scripts/interactive_learning_dashboard.py --mode $MODE

# Exit with the same status as the dashboard
exit $?
