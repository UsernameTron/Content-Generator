#!/bin/bash
# Healthcare Dashboard Test Scheduler
# Runs automated tests at specified intervals
# Usage: ./schedule_healthcare_tests.sh [options]
#
# Options:
#   -i, --interval HOURS    Time between test runs (default: 24 hours)
#   -c, --config PATH       Path to configuration file
#   -m, --mode MODE         Dashboard mode (regular or testing)
#   -d, --data-dir PATH     Path to data directory
#   -r, --report            Generate a summary report from logs
#   -h, --help              Show this help message

# Default values
INTERVAL=24
CONFIG_FILE="dashboard_config.json"
MODE="testing"
DATA_DIR="data/healthcare"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DASHBOARD_SCRIPT="${SCRIPT_DIR}/interactive_learning_dashboard.py"
TEST_SCRIPT="${SCRIPT_DIR}/run_tests.py"
LOG_DIR="${SCRIPT_DIR}/../logs"
LOG_FILE="${LOG_DIR}/scheduled_tests_$(date +%Y-%m-%d).log"
GENERATE_REPORT=false
RUN_ENHANCED_TESTS=true

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Function to generate a summary report from logs
generate_summary_report() {
    local report_file="${LOG_DIR}/performance_summary_$(date +%Y-%m-%d).txt"
    
    echo "Generating performance summary report..."
    echo "Analyzing logs in ${LOG_DIR}..."
    
    echo "=================================" > "$report_file"
    echo "  Healthcare Dashboard Performance Summary" >> "$report_file"
    echo "  Generated: $(date)" >> "$report_file"
    echo "=================================" >> "$report_file"
    echo "" >> "$report_file"
    
    # Count total test runs
    local total_runs=$(grep -l "Running scheduled tests" ${LOG_DIR}/scheduled_tests_*.log 2>/dev/null | wc -l | tr -d ' ')
    local successful_runs=$(grep -l "Tests completed successfully" ${LOG_DIR}/scheduled_tests_*.log 2>/dev/null | wc -l | tr -d ' ')
    local failed_runs=$(grep -l "Tests failed with error code" ${LOG_DIR}/scheduled_tests_*.log 2>/dev/null | wc -l | tr -d ' ')
    
    # Default to 0 if no runs found
    total_runs=${total_runs:-0}
    successful_runs=${successful_runs:-0}
    failed_runs=${failed_runs:-0}
    
    # Calculate success rate
    local success_rate=0
    if [ "$total_runs" -gt 0 ]; then
        success_rate=$((successful_runs * 100 / total_runs))
    fi
    
    echo "## Test Run Statistics" >> "$report_file"
    echo "" >> "$report_file"
    echo "- Total scheduled test runs: $total_runs" >> "$report_file"
    echo "- Successful runs: $successful_runs" >> "$report_file"
    echo "- Failed runs: $failed_runs" >> "$report_file"
    echo "- Success rate: ${success_rate}%" >> "$report_file"
    echo "" >> "$report_file"
    
    # Extract performance metrics from logs
    echo "## Performance Metrics" >> "$report_file"
    echo "" >> "$report_file"
    
    # Extract AI reasoning metrics
    grep -h "Reasoning:" ${LOG_DIR}/scheduled_tests_*.log 2>/dev/null > "${LOG_DIR}/.temp_reasoning.txt"
    grep -h "Knowledge Integration:" ${LOG_DIR}/scheduled_tests_*.log 2>/dev/null > "${LOG_DIR}/.temp_knowledge.txt"
    grep -h "Adaptability:" ${LOG_DIR}/scheduled_tests_*.log 2>/dev/null > "${LOG_DIR}/.temp_adaptability.txt"
    
    # Show AI metrics if available
    if [ -s "${LOG_DIR}/.temp_reasoning.txt" ] || [ -s "${LOG_DIR}/.temp_knowledge.txt" ] || [ -s "${LOG_DIR}/.temp_adaptability.txt" ]; then
        echo "### AI Performance Metrics" >> "$report_file"
        echo "" >> "$report_file"
        
        # Process reasoning metrics
        if [ -s "${LOG_DIR}/.temp_reasoning.txt" ]; then
            local reasoning_count=$(grep -c "Reasoning:" "${LOG_DIR}/.temp_reasoning.txt")
            local min_reasoning=$(grep -o "Reasoning: [0-9.]*" "${LOG_DIR}/.temp_reasoning.txt" | cut -d' ' -f2 | sort -n | head -1)
            local max_reasoning=$(grep -o "Reasoning: [0-9.]*" "${LOG_DIR}/.temp_reasoning.txt" | cut -d' ' -f2 | sort -nr | head -1)
            local last_reasoning=$(grep -o "Reasoning: [0-9.]*" "${LOG_DIR}/.temp_reasoning.txt" | cut -d' ' -f2 | tail -1)
            local target_reasoning=0.89
            
            echo "#### Reasoning Capability" >> "$report_file"
            echo "- Latest score: $last_reasoning (Target: $target_reasoning)" >> "$report_file"
            echo "- Range: $min_reasoning - $max_reasoning" >> "$report_file"
            echo "- Measurements: $reasoning_count" >> "$report_file"
            echo "" >> "$report_file"
        fi
        
        # Process knowledge integration metrics
        if [ -s "${LOG_DIR}/.temp_knowledge.txt" ]; then
            local knowledge_count=$(grep -c "Knowledge Integration:" "${LOG_DIR}/.temp_knowledge.txt")
            local min_knowledge=$(grep -o "Knowledge Integration: [0-9.]*" "${LOG_DIR}/.temp_knowledge.txt" | cut -d' ' -f3 | sort -n | head -1)
            local max_knowledge=$(grep -o "Knowledge Integration: [0-9.]*" "${LOG_DIR}/.temp_knowledge.txt" | cut -d' ' -f3 | sort -nr | head -1)
            local last_knowledge=$(grep -o "Knowledge Integration: [0-9.]*" "${LOG_DIR}/.temp_knowledge.txt" | cut -d' ' -f3 | tail -1)
            local target_knowledge=0.91
            
            echo "#### Knowledge Integration" >> "$report_file"
            echo "- Latest score: $last_knowledge (Target: $target_knowledge)" >> "$report_file"
            echo "- Range: $min_knowledge - $max_knowledge" >> "$report_file"
            echo "- Measurements: $knowledge_count" >> "$report_file"
            echo "" >> "$report_file"
        fi
        
        # Process adaptability metrics
        if [ -s "${LOG_DIR}/.temp_adaptability.txt" ]; then
            local adaptability_count=$(grep -c "Adaptability:" "${LOG_DIR}/.temp_adaptability.txt")
            local min_adaptability=$(grep -o "Adaptability: [0-9.]*" "${LOG_DIR}/.temp_adaptability.txt" | cut -d' ' -f2 | sort -n | head -1)
            local max_adaptability=$(grep -o "Adaptability: [0-9.]*" "${LOG_DIR}/.temp_adaptability.txt" | cut -d' ' -f2 | sort -nr | head -1)
            local last_adaptability=$(grep -o "Adaptability: [0-9.]*" "${LOG_DIR}/.temp_adaptability.txt" | cut -d' ' -f2 | tail -1)
            local target_adaptability=0.89
            
            echo "#### Adaptability" >> "$report_file"
            echo "- Latest score: $last_adaptability (Target: $target_adaptability)" >> "$report_file"
            echo "- Range: $min_adaptability - $max_adaptability" >> "$report_file"
            echo "- Measurements: $adaptability_count" >> "$report_file"
            echo "" >> "$report_file"
        fi
    fi
    
    # Extract accuracy values
    grep -h "accuracy:" ${LOG_DIR}/scheduled_tests_*.log 2>/dev/null > "${LOG_DIR}/.temp_accuracy.txt"
    
    if [ -s "${LOG_DIR}/.temp_accuracy.txt" ]; then
        # Count total accuracy measurements
        local accuracy_count=$(grep -c "accuracy:" "${LOG_DIR}/.temp_accuracy.txt")
        
        # Get min and max accuracy
        local min_accuracy=$(grep -o "accuracy: [0-9.]*" "${LOG_DIR}/.temp_accuracy.txt" | cut -d' ' -f2 | sort -n | head -1)
        local max_accuracy=$(grep -o "accuracy: [0-9.]*" "${LOG_DIR}/.temp_accuracy.txt" | cut -d' ' -f2 | sort -nr | head -1)
        
        # Get first and last accuracy for trend
        local first_accuracy=$(grep -o "accuracy: [0-9.]*" "${LOG_DIR}/.temp_accuracy.txt" | cut -d' ' -f2 | head -1)
        local last_accuracy=$(grep -o "accuracy: [0-9.]*" "${LOG_DIR}/.temp_accuracy.txt" | cut -d' ' -f2 | tail -1)
        
        echo "### Accuracy Metrics" >> "$report_file"
        echo "" >> "$report_file"
        echo "- Total accuracy measurements: $accuracy_count" >> "$report_file"
        echo "- Minimum accuracy: $min_accuracy" >> "$report_file"
        echo "- Maximum accuracy: $max_accuracy" >> "$report_file"
        echo "- First recorded accuracy: $first_accuracy" >> "$report_file"
        echo "- Latest recorded accuracy: $last_accuracy" >> "$report_file"
        echo "" >> "$report_file"
        
        # Simple trend analysis
        echo "### Performance Trend" >> "$report_file"
        echo "" >> "$report_file"
        
        # Compare first and last as integers (multiply by 100 to avoid floating point)
        local first_int=$(echo "$first_accuracy * 100" | bc | cut -d. -f1)
        local last_int=$(echo "$last_accuracy * 100" | bc | cut -d. -f1)
        
        if [ "$last_int" -gt "$first_int" ]; then
            local diff=$((last_int - first_int))
            local diff_percent=$((diff / 100)).$(printf "%02d" $((diff % 100)))
            
            if [ "$diff" -gt 5 ]; then
                echo "✅ **SIGNIFICANT IMPROVEMENT**: Accuracy increased by $diff_percent" >> "$report_file"
            else
                echo "✅ **IMPROVEMENT**: Accuracy increased by $diff_percent" >> "$report_file"
            fi
        elif [ "$first_int" -gt "$last_int" ]; then
            local diff=$((first_int - last_int))
            local diff_percent=$((diff / 100)).$(printf "%02d" $((diff % 100)))
            
            if [ "$diff" -gt 5 ]; then
                echo "⚠️ **SIGNIFICANT REGRESSION**: Accuracy decreased by $diff_percent" >> "$report_file"
            else
                echo "⚠️ **REGRESSION**: Accuracy decreased by $diff_percent" >> "$report_file"
            fi
        else
            echo "✓ **STABLE**: No significant change in accuracy" >> "$report_file"
        fi
        echo "" >> "$report_file"
    else
        echo "No accuracy metrics found in logs." >> "$report_file"
        echo "" >> "$report_file"
    fi
    
    # Extract contradiction categories
    echo "## Contradiction Detection Analysis" >> "$report_file"
    echo "" >> "$report_file"
    
    grep -h "category:" ${LOG_DIR}/scheduled_tests_*.log 2>/dev/null > "${LOG_DIR}/.temp_categories.txt"
    
    if [ -s "${LOG_DIR}/.temp_categories.txt" ]; then
        echo "### Contradiction Categories" >> "$report_file"
        echo "" >> "$report_file"
        
        # Count occurrences of each category
        grep -o "category: [a-zA-Z_]*" "${LOG_DIR}/.temp_categories.txt" | 
            sort | uniq -c | sort -nr | 
            while read count category; do
                category=${category#category: }
                echo "- $category: $count occurrences" >> "$report_file"
            done
        echo "" >> "$report_file"
    else
        echo "No contradiction categories found in logs." >> "$report_file"
        echo "" >> "$report_file"
    fi
    
    # Extract error patterns
    echo "## Error Analysis" >> "$report_file"
    echo "" >> "$report_file"
    
    grep -h "Error:" ${LOG_DIR}/scheduled_tests_*.log 2>/dev/null > "${LOG_DIR}/.temp_errors.txt"
    
    if [ -s "${LOG_DIR}/.temp_errors.txt" ]; then
        echo "### Top Error Patterns" >> "$report_file"
        echo "" >> "$report_file"
        
        # Count occurrences of each error
        sort "${LOG_DIR}/.temp_errors.txt" | uniq -c | sort -nr | head -5 |
            while read count error; do
                echo "- $error ($count occurrences)" >> "$report_file"
            done
        echo "" >> "$report_file"
    else
        echo "No error patterns found in logs." >> "$report_file"
        echo "" >> "$report_file"
    fi
    
    # Extract enhanced metrics (Customer Experience and AI)
    echo "## Enhanced Metrics Analysis" >> "$report_file"
    echo "" >> "$report_file"
    
    # Extract Customer Experience metrics
    grep -h "Customer Experience Results" -A10 ${LOG_DIR}/scheduled_tests_*.log 2>/dev/null > "${LOG_DIR}/.temp_ce.txt"
    
    if [ -s "${LOG_DIR}/.temp_ce.txt" ]; then
        echo "### Customer Experience Metrics" >> "$report_file"
        echo "" >> "$report_file"
        
        # Extract overall score and get latest value
        local ce_score=$(grep -h "Overall Score" "${LOG_DIR}/.temp_ce.txt" | tail -1 | awk '{print $NF}')
        local ce_baseline="0.81"
        local ce_target="0.91"
        
        echo "- Current Score: $ce_score (Baseline: $ce_baseline, Target: $ce_target)" >> "$report_file"
        
        # Calculate progress percentage
        if [ -n "$ce_score" ]; then
            local ce_progress=$(echo "scale=2; ($ce_score - $ce_baseline) / ($ce_target - $ce_baseline) * 100" | bc)
            echo "- Progress toward target: ${ce_progress}%" >> "$report_file"
        fi
        echo "" >> "$report_file"
    else
        echo "No Customer Experience metrics found in logs." >> "$report_file"
        echo "" >> "$report_file"
    fi
    
    # Extract Artificial Intelligence metrics
    grep -h "Artificial Intelligence Results" -A10 ${LOG_DIR}/scheduled_tests_*.log 2>/dev/null > "${LOG_DIR}/.temp_ai.txt"
    
    if [ -s "${LOG_DIR}/.temp_ai.txt" ]; then
        echo "### Artificial Intelligence Metrics" >> "$report_file"
        echo "" >> "$report_file"
        
        # Extract overall score and get latest value
        local ai_score=$(grep -h "Overall Score" "${LOG_DIR}/.temp_ai.txt" | tail -1 | awk '{print $NF}')
        local ai_baseline="0.76"
        local ai_target="0.88"
        
        echo "- Current Score: $ai_score (Baseline: $ai_baseline, Target: $ai_target)" >> "$report_file"
        
        # Calculate progress percentage
        if [ -n "$ai_score" ]; then
            local ai_progress=$(echo "scale=2; ($ai_score - $ai_baseline) / ($ai_target - $ai_baseline) * 100" | bc)
            echo "- Progress toward target: ${ai_progress}%" >> "$report_file"
        fi
        echo "" >> "$report_file"
    else
        echo "No Artificial Intelligence metrics found in logs." >> "$report_file"
        echo "" >> "$report_file"
    fi
    
    # Generate ASCII chart for performance trend
    echo "## Performance Timeline" >> "$report_file"
    echo "" >> "$report_file"
    
    # Extract accuracy values for chart
    grep -o "accuracy: [0-9.]*" "${LOG_DIR}/scheduled_tests_*.log" 2>/dev/null |
        cut -d' ' -f2 > "${LOG_DIR}/.temp_chart.txt"
    
    if [ -s "${LOG_DIR}/.temp_chart.txt" ]; then
        echo "```" >> "$report_file"
        echo "Accuracy Trend (last 10 runs)" >> "$report_file"
        echo "" >> "$report_file"
        
        # Create simple ASCII chart (last 10 entries)
        tail -n 10 "${LOG_DIR}/.temp_chart.txt" | cat -n |
            while read idx accuracy; do
                # Convert accuracy to bar length (0.0-1.0 to 0-40 characters)
                local bar_length=$(echo "$accuracy * 40" | bc | cut -d. -f1)
                local bar=$(printf '%0.s#' $(seq 1 $bar_length))
                printf "Run %2d [%-40s] %s\n" "$idx" "$bar" "$accuracy" >> "$report_file"
            done
        
        echo "```" >> "$report_file"
        echo "" >> "$report_file"
    fi
    
    # Add recommendations based on analysis
    echo "## Recommendations" >> "$report_file"
    echo "" >> "$report_file"
    
    # Simple recommendations based on trend
    if [ "$last_int" -lt "$first_int" ]; then
        echo "1. **Investigate Performance Regression**: Review recent changes that may have caused accuracy decline" >> "$report_file"
        echo "2. **Increase Test Frequency**: Consider running tests more frequently to catch issues earlier" >> "$report_file"
        echo "3. **Review Error Patterns**: Address the most common errors identified in the logs" >> "$report_file"
    elif [ "$last_int" -gt "$first_int" ]; then
        echo "1. **Maintain Improvement Trajectory**: Continue with current approach as it's showing positive results" >> "$report_file"
        echo "2. **Document Effective Changes**: Record what changes led to the improvement for future reference" >> "$report_file"
        echo "3. **Consider Further Optimizations**: Explore additional enhancements to build on current success" >> "$report_file"
    else
        echo "1. **Diversify Test Cases**: Add more varied test scenarios to challenge the system" >> "$report_file"
        echo "2. **Review Configuration**: Experiment with different configuration parameters to improve performance" >> "$report_file"
        echo "3. **Optimize Resource Usage**: Check if resource constraints are limiting performance improvement" >> "$report_file"
    fi
    
    # Clean up temporary files
    rm -f "${LOG_DIR}/.temp_accuracy.txt" "${LOG_DIR}/.temp_categories.txt" "${LOG_DIR}/.temp_errors.txt" "${LOG_DIR}/.temp_chart.txt"
    
    echo "Summary report generated: $report_file"
    cat "$report_file"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -i|--interval)
      INTERVAL="$2"
      shift 2
      ;;
    -c|--config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    -m|--mode)
      MODE="$2"
      shift 2
      ;;
    -d|--data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    -r|--report)
      GENERATE_REPORT=true
      shift
      ;;
    --no-enhanced-tests)
      RUN_ENHANCED_TESTS=false
      shift
      ;;
    -h|--help)
      echo "Healthcare Dashboard Test Scheduler"
      echo "Usage: ./schedule_healthcare_tests.sh [options]"
      echo ""
      echo "Options:"
      echo "  -i, --interval HOURS    Time between test runs (default: 24 hours)"
      echo "  -c, --config PATH       Path to configuration file"
      echo "  -m, --mode MODE         Dashboard mode (regular or testing)"
      echo "  -d, --data-dir PATH     Path to data directory"
      echo "  -r, --report            Generate a summary report from logs"
      echo "  --no-enhanced-tests     Skip running enhanced tests"
      echo "  -h, --help              Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Generate report if requested
if [ "$GENERATE_REPORT" = true ]; then
  generate_summary_report
  exit 0
fi

# Validate scripts exist
if [ ! -f "$DASHBOARD_SCRIPT" ]; then
  echo "Error: Dashboard script not found at $DASHBOARD_SCRIPT"
  exit 1
fi

if [ "$RUN_ENHANCED_TESTS" = true ] && [ ! -f "$TEST_SCRIPT" ]; then
  echo "Error: Test script not found at $TEST_SCRIPT"
  exit 1
fi

# Print configuration
echo "Healthcare Dashboard Test Scheduler"
echo "=================================="
echo "Interval:       Every $INTERVAL hours"
echo "Config File:    $CONFIG_FILE"
echo "Dashboard Mode: $MODE"
echo "Data Directory: $DATA_DIR"
echo "Log File:       $LOG_FILE"
echo "Enhanced Tests: $RUN_ENHANCED_TESTS"
echo "=================================="
echo "Starting scheduler at $(date)"
echo "Press Ctrl+C to stop"
echo ""

# Main loop
while true; do
  TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
  echo "[$TIMESTAMP] Running scheduled tests..." | tee -a "$LOG_FILE"
  
  # Run the dashboard with scheduled test flag
  cd "$SCRIPT_DIR"
  python3 "$DASHBOARD_SCRIPT" --run-scheduled-tests --mode "$MODE" --config "$CONFIG_FILE" --data-dir "$DATA_DIR" 2>&1 | tee -a "$LOG_FILE"
  
  # Check if the command succeeded
  if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "[$TIMESTAMP] Dashboard tests completed successfully" | tee -a "$LOG_FILE"
  else
    echo "[$TIMESTAMP] Dashboard tests failed with error code ${PIPESTATUS[0]}" | tee -a "$LOG_FILE"
  fi
  
  # Run enhanced tests if enabled
  if [ "$RUN_ENHANCED_TESTS" = true ]; then
    echo "[$TIMESTAMP] Running enhanced tests for Customer Experience and AI metrics..." | tee -a "$LOG_FILE"
    
    # Create timestamp-based output file for enhanced tests
    local test_output="${SCRIPT_DIR}/../reports/tests/enhanced_tests_$(date +%Y%m%d_%H%M%S).json"
    
    # Run enhanced tests
    python3 "$TEST_SCRIPT" --output "$test_output" --preset "$MODE" 2>&1 | tee -a "$LOG_FILE"
    
    # Check if the command succeeded
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
      echo "[$TIMESTAMP] Enhanced tests completed successfully" | tee -a "$LOG_FILE"
    else
      echo "[$TIMESTAMP] Enhanced tests failed with error code ${PIPESTATUS[0]}" | tee -a "$LOG_FILE"
    fi
  fi
  
  # Generate performance report
  if [ $(( RANDOM % 5 )) -eq 0 ]; then
    echo "[$TIMESTAMP] Generating periodic performance summary..." | tee -a "$LOG_FILE"
    generate_summary_report
  fi
  
  # Calculate next run time
  NEXT_RUN=$(date -v+${INTERVAL}H "+%Y-%m-%d %H:%M:%S")
  echo "[$TIMESTAMP] Next run scheduled for $NEXT_RUN" | tee -a "$LOG_FILE"
  echo "Sleeping for $INTERVAL hours..." | tee -a "$LOG_FILE"
  echo "" | tee -a "$LOG_FILE"
  
  # Sleep until next run
  sleep ${INTERVAL}h
done
