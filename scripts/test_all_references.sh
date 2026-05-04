#!/bin/bash
# Test all references one by one, recording results to a file.
# Prefer using test_all_references.py instead (supports CSV + resume).
set -u

cd /home/li004074/CUDA-Hercules
RESULTS_FILE="results/reference_test_results.txt"
mkdir -p results

> "$RESULTS_FILE"

echo "Testing all references..."
echo "Results will be written to $RESULTS_FILE"

PASS=0
FAIL=0
TOTAL=0

for task_dir in $(find tasks/ -name def.py -exec dirname {} \; | sort); do
    task_name=$(basename "$task_dir")
    if [ ! -f "$task_dir/reference.cu" ]; then
        echo "$task_name: SKIP (no reference.cu)" | tee -a "$RESULTS_FILE"
        continue
    fi

    TOTAL=$((TOTAL + 1))
    echo -n "[$TOTAL] Testing $task_name ... "

    OUTPUT=$(python scripts/test_reference.py "$task_dir" 2>&1)

    if echo "$OUTPUT" | grep -q "PASS$"; then
        echo "PASS"
        echo "$task_name: PASS" >> "$RESULTS_FILE"
        PASS=$((PASS + 1))
    else
        echo "FAIL"
        echo "$task_name: FAIL" >> "$RESULTS_FILE"
        echo "--- $task_name DETAILS ---" >> "$RESULTS_FILE"
        echo "$OUTPUT" | tail -30 >> "$RESULTS_FILE"
        echo "--- END ---" >> "$RESULTS_FILE"
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "============================================"
echo "SUMMARY: $PASS PASS, $FAIL FAIL, $TOTAL TOTAL"
echo "============================================"
echo "" >> "$RESULTS_FILE"
echo "SUMMARY: $PASS PASS, $FAIL FAIL, $TOTAL TOTAL" >> "$RESULTS_FILE"
