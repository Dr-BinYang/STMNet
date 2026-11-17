#!/usr/bin/env bash
set -Eeuo pipefail

# Download Traffic for STMNet.
# Source: https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/traffic/traffic.csv

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$SCRIPT_DIR"
DEST_DIR="$DATASET_DIR/traffic"
DEST_FILE="$DEST_DIR/traffic.csv"
URL="https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/traffic/traffic.csv"
EXPECTED_COLUMNS="863"
EXPECTED_SHA256="cb06463d56fa17d87f47027cd9389ceae82a69eddee51cdb61480e120dab0b16"

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

validate_csv() {
    local file="$1"
    local columns="$2"

    awk -F',' -v expected="$columns" '
        NR == 1 {
            sub(/\r$/, "", $1)
            if ($1 != "date" || NF != expected) exit 1
        }
        NR == 2 { has_data = 1 }
        END {
            if (NR < 2 || !has_data) exit 1
        }
    ' "$file"
}

verify_sha256() {
    local file="$1"
    local expected="$2"
    local actual=""

    [[ -z "$expected" ]] && return 0

    if command -v sha256sum >/dev/null 2>&1; then
        actual="$(sha256sum "$file" | awk '{print $1}')"
    elif command -v shasum >/dev/null 2>&1; then
        actual="$(shasum -a 256 "$file" | awk '{print $1}')"
    else
        echo "Warning: no SHA-256 utility found; skipping checksum verification."
        return 0
    fi

    if [[ "$actual" != "$expected" ]]; then
        echo "Error: SHA-256 mismatch for $file" >&2
        echo "Expected: $expected" >&2
        echo "Actual:   $actual" >&2
        return 1
    fi
}

mkdir -p "$DEST_DIR"

if [[ -s "$DEST_FILE" ]] && validate_csv "$DEST_FILE" "$EXPECTED_COLUMNS" \
    && verify_sha256 "$DEST_FILE" "$EXPECTED_SHA256"; then
    echo "Traffic already exists and passed validation:"
    echo "$DEST_FILE"
    exit 0
fi

TMP_FILE="${DEST_FILE}.part"
rm -f "$TMP_FILE"

echo "Downloading Traffic..."
download_file "$URL" "$TMP_FILE"

if [[ ! -s "$TMP_FILE" ]]; then
    echo "Error: downloaded file is empty." >&2
    rm -f "$TMP_FILE"
    exit 1
fi

if ! validate_csv "$TMP_FILE" "$EXPECTED_COLUMNS"; then
    echo "Error: CSV validation failed (expected $EXPECTED_COLUMNS columns and a date header)." >&2
    rm -f "$TMP_FILE"
    exit 1
fi

if ! verify_sha256 "$TMP_FILE" "$EXPECTED_SHA256"; then
    rm -f "$TMP_FILE"
    exit 1
fi

mv -f "$TMP_FILE" "$DEST_FILE"
echo "Successfully downloaded Traffic:"
echo "$DEST_FILE"
