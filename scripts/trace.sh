#!/usr/bin/env bash
set -euo pipefail

TYPE="${1:?Usage: $0 <bin|example> <target>}"
TARGET="${2:?Usage: $0 <bin|example> <target>}"

case "$TYPE" in
    bin)
        cargo run --release --features trace --bin "$TARGET"
        ;;
    example)
        cargo run --release --features trace --example "$TARGET"
        ;;
    *)
        echo "Unknown target type: $TYPE (expected bin or example)"
        exit 1
        ;;
esac