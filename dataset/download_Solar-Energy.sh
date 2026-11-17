#!/usr/bin/env bash
set -Eeuo pipefail

# Download the Solar-Energy (Alabama PV) dataset for STMNet.
# Original benchmark source:
# https://github.com/laiguokun/multivariate-time-series-data
# The source archive is solar_AL.txt.gz. It is decompressed to solar_AL.txt.
# MD5 checksum of the source archive is documented by torch_timeseries:
# 41ef7fdc958c2ca3fac9cd06d6227073

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="$SCRIPT_DIR/solar"
DEST_FILE="$DEST_DIR/solar_AL.txt"
URL="https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/solar-energy/solar_AL.txt.gz"
EXPECTED_MD5="41ef7fdc958c2ca3fac9cd06d6227073"

download_file() {
    local url="$1"
    local output="$2"

    if command -v curl >/dev/null 2>&1; then
        curl --fail --location --show-error --retry 5 --retry-all-errors \
            --retry-delay 2 --connect-timeout 30 --output "$output" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget --tries=5 --timeout=30 --output-document="$output" "$url"
    else
        echo "Error: curl or wget is required." >&2
        return 1
    fi
}

validate_solar() {
    local file="$1"

    awk -F',' '
        NR == 1 {
            if (NF != 137) exit 1
            for (i = 1; i <= NF; i++) {
                if ($i !~ /^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/) exit 1
            }
        }
        NR == 2 { has_data = 1 }
        END {
            if (NR < 2 || !has_data) exit 1
        }
    ' "$file"
}

verify_md5() {
    local file="$1"
    local actual=""

    if command -v md5sum >/dev/null 2>&1; then
        actual="$(md5sum "$file" | awk '{print $1}')"
    elif command -v md5 >/dev/null 2>&1; then
        actual="$(md5 -q "$file")"
    else
        echo "Warning: no MD5 utility found; skipping checksum verification."
        return 0
    fi

    if [[ "$actual" != "$EXPECTED_MD5" ]]; then
        echo "Error: MD5 mismatch for Solar-Energy archive." >&2
        echo "Expected: $EXPECTED_MD5" >&2
        echo "Actual:   $actual" >&2
        return 1
    fi
}

mkdir -p "$DEST_DIR"

if [[ -s "$DEST_FILE" ]] && validate_solar "$DEST_FILE"; then
    echo "Solar-Energy already exists and passed validation:"
    echo "$DEST_FILE"
    exit 0
fi

GZ_TMP="$DEST_FILE.gz.part"
TXT_TMP="$DEST_FILE.part"
rm -f "$GZ_TMP" "$TXT_TMP"

echo "Downloading Solar-Energy..."
download_file "$URL" "$GZ_TMP"

if [[ ! -s "$GZ_TMP" ]] || ! gzip -t "$GZ_TMP"; then
    echo "Error: downloaded Solar-Energy archive is empty or invalid." >&2
    rm -f "$GZ_TMP" "$TXT_TMP"
    exit 1
fi

if ! verify_md5 "$GZ_TMP"; then
    rm -f "$GZ_TMP" "$TXT_TMP"
    exit 1
fi

gzip -dc "$GZ_TMP" > "$TXT_TMP"

if ! validate_solar "$TXT_TMP"; then
    echo "Error: Solar-Energy data validation failed (expected 137 numeric columns)." >&2
    rm -f "$GZ_TMP" "$TXT_TMP"
    exit 1
fi

mv -f "$TXT_TMP" "$DEST_FILE"
rm -f "$GZ_TMP"

echo "Successfully downloaded and extracted Solar-Energy:"
echo "$DEST_FILE"
