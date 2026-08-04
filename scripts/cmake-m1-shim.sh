#!/usr/bin/env bash
# Workaround for a nasty toolchain wedge (found 2026-08-04): ggml's cmake
# feature detection EXECUTES ARM instructions (i8mm/sve/sme probes) to see if
# they work. On this M1 + current macOS build, running an unsupported
# instruction makes the probe binary spin forever instead of dying with
# SIGILL — wedging every whisper-rs configure (and, previously, `tauri
# build`). Presetting the probe results skips the runs entirely.
# dotprod is genuinely supported on M1; i8mm/sve/sme are not.
set -euo pipefail
REAL_CMAKE="${DISTIL_REAL_CMAKE:-/opt/homebrew/bin/cmake}"
for arg in "$@"; do
  case "$arg" in
    --build|--install|--version|-E|-P) exec "$REAL_CMAKE" "$@" ;;
  esac
done
exec "$REAL_CMAKE" "$@" \
  -DGGML_MACHINE_SUPPORTS_dotprod=1 \
  -DGGML_MACHINE_SUPPORTS_i8mm=0 \
  -DGGML_MACHINE_SUPPORTS_sve=0 \
  -DGGML_MACHINE_SUPPORTS_sme=0
