#!/usr/bin/env bash
# Build lambda.zip with Linux/arm64 wheels (Lambda runtime), regardless of your laptop OS.
set -euo pipefail
cd "$(dirname "$0")"
rm -rf build lambda.zip && mkdir -p build
pip install --quiet --target build --platform manylinux2014_aarch64 \
  --implementation cp --python-version 3.13 --only-binary=:all: \
  "anthropic>=0.100" "openai>=1.40" "fastapi>=0.110" "pydantic>=2.6" "mangum>=0.17"
cp -r src/finops_agent build/
(cd build && zip -q -r ../lambda.zip . -x "*.pyc" "*/__pycache__/*")
echo "built lambda.zip ($(du -h lambda.zip | cut -f1))"
