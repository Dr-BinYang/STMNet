# STMNet Dataset Download Scripts

This directory contains independent download scripts for the nine datasets used in STMNet:

- ETTm1
- ETTm2
- ETTh1
- ETTh2
- Exchange
- Weather
- Solar-Energy
- Electricity
- Traffic

The scripts download the datasets from their public source repositories. 

## Requirements

- Bash
- `curl` or `wget`
- `gzip` for the Solar-Energy archive
- `sha256sum` or `shasum` for SHA-256 verification where a checksum is configured
- `md5sum` or `md5` for Solar-Energy archive verification

On Windows, run these scripts using WSL, Git Bash, or another Bash-compatible environment.

## Download a Single Dataset

Run the corresponding script from the STMNet repository root. For example:

```bash
bash dataset/download_ETTm1.sh
bash dataset/download_ETTm2.sh
bash dataset/download_ETTh1.sh
bash dataset/download_ETTh2.sh
bash dataset/download_Exchange.sh
bash dataset/download_Weather.sh
bash dataset/download_Solar-Energy.sh
bash dataset/download_Electricity.sh
bash dataset/download_Traffic.sh
```

Each script creates its destination directory automatically. Existing files are checked before downloading again.

## Download All Datasets

To download all nine datasets sequentially:

```bash
bash dataset/download_all.sh
```

If a download fails, the script stops and reports the failing dataset. Fix the network or source issue, then rerun the command. Successfully downloaded datasets that pass validation will be skipped.

## Downloaded File Layout

After all downloads complete, the directory structure will be:

```text
dataset/
├── ETT-small/
│   ├── ETTm1.csv
│   ├── ETTm2.csv
│   ├── ETTh1.csv
│   └── ETTh2.csv
│
├── exchange_rate/
│   └── exchange_rate.csv
│
├── weather/
│   └── weather.csv
│
├── solar/
│   └── solar_AL.txt
│
├── electricity/
│   └── electricity.csv
│
└── traffic/
    └── traffic.csv
```

## Dataset Sources

The ETT, Exchange, Weather, Electricity, and Traffic scripts use files hosted by the Hugging Face mirror of Time-Series-Library:

- [Time-Series-Library dataset repository](https://huggingface.co/datasets/thuml/Time-Series-Library)

The Solar-Energy script downloads the compressed `solar_AL.txt.gz` file from the multivariate time-series dataset repository and extracts it to `solar/solar_AL.txt`:

- [Multivariate Time-Series Dataset repository](https://github.com/laiguokun/multivariate-time-series-data)


