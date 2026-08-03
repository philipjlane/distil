#!/usr/bin/env bash
# Builds the Swift capture sidecar and drops it where Tauri expects external
# binaries: src-tauri/binaries/distil-capture-<target-triple>.
set -euo pipefail

cd "$(dirname "$0")/../src-capture"
swift build -c release

ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
  TRIPLE="aarch64-apple-darwin"
else
  TRIPLE="x86_64-apple-darwin"
fi

mkdir -p ../src-tauri/binaries
cp .build/release/distil-capture "../src-tauri/binaries/distil-capture-${TRIPLE}"
echo "sidecar -> src-tauri/binaries/distil-capture-${TRIPLE}"
