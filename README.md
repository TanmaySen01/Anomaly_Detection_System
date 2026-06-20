# Anomaly Detection System

A generalized, reusable Python tool for detecting anomalies in any time-series or sensor dataset using Isolation Forest. Built while preparing for an ML internship at DRDO Gwalior, with the goal of being applicable across domains — industrial sensors, network traffic, server monitoring, and beyond.

## What it does

Most monitoring systems rely on fixed thresholds (e.g. "alert if temperature > 90°C"), which miss two common failure patterns:

- Gradual drift that never crosses a threshold but signals a developing problem
- Context-dependent values — a high reading that's good in one scenario (internet speed) and bad in another (machine temperature)

This tool instead learns what "normal" looks like directly from the data, using the Isolation Forest algorithm, and flags points that don't fit that pattern — without needing manually tuned thresholds for every use case.

## Features

- Works with **any CSV file** — local or hosted online
- Fully **interactive** — no code editing required, just run and answer prompts
- Automatically handles **missing values**
- Supports **multiple features at once** (e.g. temperature + network traffic together)
- Generates a **visualization** with anomalies clearly marked

## Tech stack

- Python
- pandas, numpy
- scikit-learn (Isolation Forest)
- matplotlib

. Answer the prompts:
   - CSV file path or URL
   - Which column(s) to check for anomalies
   - Expected anomaly percentage (or press Enter for the default)
   - Column to use for the X-axis (or press Enter to use row number)

## Tested on

- **Real industrial machine temperature data** (NAB benchmark dataset) — correctly identified two real equipment failure events
- **AWS EC2 CPU utilization data** — correctly flagged unusual usage spikes on a completely different dataset, with zero code changes

## Future improvements

- Add time-aware detection to distinguish "rare but routine" events (e.g. expected daily spikes) from genuinely unexpected anomalies
- Add an Autoencoder-based detection option for comparison against Isolation Forest
- Combine machine sensor data with network traffic data for joint anomaly detection

## Author

Tanmay Sen
