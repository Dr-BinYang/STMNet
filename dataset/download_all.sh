#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

SCRIPTS=(
    download_ETTm1.sh
    download_ETTm2.sh
    download_ETTh1.sh
    download_ETTh2.sh
    download_Exchange.sh
    download_Weather.sh
    download_Solar-Energy.sh
    download_Electricity.sh
    download_Traffic.sh
)

for script in "${SCRIPTS[@]}"; do
    echo
    echo "============================================================"
    echo "Running ${script}"
    echo "============================================================"
    bash "$SCRIPT_DIR/$script"
done

echo
echo "All STMNet datasets have been downloaded and validated."
