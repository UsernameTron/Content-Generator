#!/bin/bash
# Desktop launcher for healthcare tests
# This script provides an interactive menu to run various healthcare tests

# Set up constants
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="${BASE_DIR}/scripts"
TEST_SCRIPT="${SCRIPTS_DIR}/run_tests.py"
SCHEDULER_SCRIPT="${SCRIPTS_DIR}/schedule_healthcare_tests.sh"

# Function to display menu
display_menu() {
  clear
  echo "==============================================="
  echo "    Healthcare Performance Metrics Testing     "
  echo "==============================================="
  echo "1. Run Comprehensive Tests (All Metrics)"
  echo "2. Run Customer Experience Tests"
  echo "3. Run Artificial Intelligence Tests"
  echo "4. Schedule Automated Tests (Every 24 Hours)"
  echo "5. Generate Summary Report from Logs"
  echo "6. View Latest Test Results"
  echo "7. Exit"
  echo "==============================================="
  echo -n "Select an option [1-7]: "
}

# Function to run tests with specified preset
run_tests() {
  local preset=$1
  local output_file="${BASE_DIR}/reports/tests/test_results_${preset}_$(date +%Y%m%d_%H%M%S).json"
  
  echo "Running tests with preset: $preset"
  echo "Output will be saved to: $output_file"
  echo ""
  
  cd "$SCRIPTS_DIR"
  python3 "$TEST_SCRIPT" --preset "$preset" --output "$output_file"
  
  if [ $? -eq 0 ]; then
    echo ""
    echo "Tests completed successfully!"
    echo "Results saved to: $output_file"
  else
    echo ""
    echo "Tests failed with error code $?"
  fi
  
  echo ""
  read -p "Press Enter to continue..."
}

# Function to view the latest test result
view_latest_results() {
  # Find the latest JSON test result file
  local latest_file=$(find "${BASE_DIR}/reports/tests" -name "*.json" -type f -print0 | xargs -0 ls -t | head -1)
  
  if [ -z "$latest_file" ]; then
    echo "No test results found."
    read -p "Press Enter to continue..."
    return
  fi
  
  echo "Displaying latest test results from: $latest_file"
  echo ""
  
  # Use jq to pretty-print the JSON if available, otherwise use cat
  if command -v jq >/dev/null 2>&1; then
    jq . "$latest_file" | less
  else
    less "$latest_file"
  fi
}

# Check prerequisites
if [ ! -f "$TEST_SCRIPT" ]; then
  echo "Error: Test script not found at $TEST_SCRIPT"
  exit 1
fi

# Make sure the reports directory exists
mkdir -p "${BASE_DIR}/reports/tests"

# Main loop
while true; do
  display_menu
  read choice
  
  case $choice in
    1)
      run_tests "comprehensive"
      ;;
    2)
      run_tests "customer_experience"
      ;;
    3)
      run_tests "artificial_intelligence"
      ;;
    4)
      echo ""
      echo "Starting automated test scheduler in a new terminal window..."
      echo "Press Ctrl+C in that window to stop the scheduler."
      osascript -e "tell application \"Terminal\" to do script \"cd '$BASE_DIR' && bash '$SCHEDULER_SCRIPT'\""
      echo ""
      read -p "Press Enter to continue..."
      ;;
    5)
      echo ""
      echo "Generating performance summary report from logs..."
      bash "$SCHEDULER_SCRIPT" --report
      echo ""
      read -p "Press Enter to continue..."
      ;;
    6)
      view_latest_results
      ;;
    7)
      echo "Exiting..."
      exit 0
      ;;
    *)
      echo "Invalid option. Please try again."
      sleep 1
      ;;
  esac
done
