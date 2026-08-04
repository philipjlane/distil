#!/usr/bin/env bash
# Build a runnable Distil.app: tauri build + the sherpa dylib fixup.
#
# sherpa-rs (download-binaries) links onnxruntime + sherpa-onnx-c-api as
# DYNAMIC libraries dropped next to the binary in target/<profile>/ — the
# bundled .app has neither the dylibs nor an rpath, so it dies at dyld.
# This script copies them into Contents/Frameworks, adds the rpath, and
# re-signs (ad-hoc). Usage:
#
#   scripts/bundle-app.sh            # build + fix up
#   scripts/bundle-app.sh --install  # …and install to /Applications + launch
set -euo pipefail
cd "$(dirname "$0")/.."

export PATH="$HOME/.cargo/bin:/opt/homebrew/bin:$PATH"
npm run tauri build

APP=src-tauri/target/release/bundle/macos/Distil.app
FRAMEWORKS="$APP/Contents/Frameworks"
mkdir -p "$FRAMEWORKS"
cp src-tauri/target/release/libonnxruntime.*.dylib \
   src-tauri/target/release/libsherpa-onnx-c-api.dylib \
   "$FRAMEWORKS/"
install_name_tool -add_rpath "@executable_path/../Frameworks" \
  "$APP/Contents/MacOS/distil" 2>/dev/null || true  # ok if already present
codesign --force --deep -s - "$APP"
echo "bundled: $APP"

if [ "${1:-}" = "--install" ]; then
  osascript -e 'quit app "Distil"' 2>/dev/null || true
  sleep 1
  rm -rf /Applications/Distil.app
  ditto "$APP" /Applications/Distil.app
  open -a /Applications/Distil.app
  echo "installed and launched /Applications/Distil.app"
fi
