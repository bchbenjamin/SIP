#!/usr/bin/env bash
# Build contract script for SIP Guardian Backend
# Compiles, tests, and places the production JAR at database/backend/sip-backend-0.1.0.jar
# which is consumed directly by the Dockerfile for Render and container deployments.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BACKEND_DIR="${REPO_ROOT}/database/backend"

echo "=== [1/3] Building and testing SIP Backend with Gradle ==="
cd "${BACKEND_DIR}"
./gradlew --no-daemon clean test bootJar

echo "=== [2/3] Synchronizing JAR artifact to canonical Docker build location ==="
cp "${BACKEND_DIR}/build/libs/sip-backend-0.1.0.jar" "${BACKEND_DIR}/sip-backend-0.1.0.jar"

echo "=== [3/3] Verification of produced artifact ==="
sha256sum "${BACKEND_DIR}/sip-backend-0.1.0.jar"
echo "SIP Backend JAR build contract satisfied."
