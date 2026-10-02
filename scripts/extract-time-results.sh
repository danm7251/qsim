#!/usr/bin/env bash
set -euo pipefail

root=${1:-target/criterion}
out=${2:-target/criterion/results.csv}

echo "group,function,param,mean_ns,mean_ci_lo_ns,mean_ci_hi_ns,std_dev_ns,samples" > "$out"

find "$root" -path '*/new/estimates.json' | sort | while read -r f; do
    dir=${f%/new/estimates.json}
    name=${dir#"$root"/}
    n=$(jq '.iters | length' "$dir/new/sample.json")
    jq -r --arg name "$name" --argjson n "$n" '
        ($name | split("/")) as $p
        | [ ($p[0] // ""), ($p[1] // ""), ($p[2:] | join("/")),
            .mean.point_estimate,
            .mean.confidence_interval.lower_bound,
            .mean.confidence_interval.upper_bound,
            .std_dev.point_estimate,
            $n ] | @csv' "$f"
done >> "$out"