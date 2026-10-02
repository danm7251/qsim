#!/usr/bin/env bash
# Collects every DHAT JSON result under target/dhat/ into a single CSV.
#
# Usage: ./dhat_to_csv.sh [INPUT_DIR] [OUTPUT_CSV]
# Defaults: INPUT_DIR=target/dhat, OUTPUT_CSV=INPUT_DIR/results.csv
#
# Columns: group, case, label, n, peak_bytes, total_allocations

set -euo pipefail

input_dir="${1:-target/dhat}"
output="${2:-$input_dir/results.csv}"

[ -d "$input_dir" ] || { echo "Input directory not found: $input_dir" >&2; exit 1; }

echo "group,case,label,n,peak_bytes,total_allocations" > "$output"

for file in "$input_dir"/*/*.json; do
    [ -e "$file" ] || continue
    group=$(basename "$(dirname "$file")")
    case_name=$(basename "$file" .json)

    jq -r --arg group "$group" --arg case "$case_name" '
        def sum(k): [.pps[]? | .[k] // 0] | add // 0;
        ($case | capture("^(?<label>.*)-(?<n>[0-9]+)$") // {label: $case, n: ""}) as $c
        | [$group, $case, $c.label, $c.n, sum("gb"), sum("tbk")]
        | @csv
    ' "$file" >> "$output"
done

# Sort the data rows by group, label and numeric n, keeping the header first.
{
    head -n 1 "$output"
    tail -n +2 "$output" | sort -t, -k1,1 -k3,3 -k4,4n
} > "$output.tmp"
mv "$output.tmp" "$output"

echo "Wrote $(($(wc -l < "$output") - 1)) rows to $output"