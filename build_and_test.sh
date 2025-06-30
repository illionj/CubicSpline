#!/usr/bin/env bash
rm -r build
set -euo pipefail

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
cd build/bundle/
./demo
python3 ./scripts/show.py 

