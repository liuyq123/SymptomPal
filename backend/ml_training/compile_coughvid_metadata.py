#!/usr/bin/env python3
"""
Compile COUGHVID JSON metadata files into a single CSV.
"""

import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def compile_metadata(data_dir: str, output_file: str):
    """Compile all JSON metadata files into a CSV."""
    data_path = Path(data_dir)
    json_files = list(data_path.glob("*.json"))

    print(f"Found {len(json_files)} JSON files")

    records = []
    for json_file in tqdm(json_files, desc="Processing"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Extract UUID from filename
            uuid = json_file.stem

            # Check for corresponding audio file
            audio_ext = None
            for ext in ['.webm', '.ogg', '.wav']:
                if (data_path / f"{uuid}{ext}").exists():
                    audio_ext = ext
                    break

            record = {
                'uuid': uuid,
                'datetime': data.get('datetime'),
                'cough_detected': float(data.get('cough_detected', 0)),
                'latitude': data.get('latitude'),
                'longitude': data.get('longitude'),
                'age': data.get('age'),
                'gender': data.get('gender'),
                'respiratory_condition': data.get('respiratory_condition'),
                'fever_muscle_pain': data.get('fever_muscle_pain'),
                'status': data.get('status'),
            }

            # Extract expert labels (up to 3 experts)
            for i in range(1, 4):
                expert_key = f'expert_labels_{i}'
                if expert_key in data:
                    expert = data[expert_key]
                    record[f'quality_{i}'] = expert.get('quality')
                    record[f'cough_type_{i}'] = expert.get('cough_type')
                    record[f'dyspnea_{i}'] = expert.get('dyspnea')
                    record[f'wheezing_{i}'] = expert.get('wheezing')
                    record[f'stridor_{i}'] = expert.get('stridor')
                    record[f'choking_{i}'] = expert.get('choking')
                    record[f'congestion_{i}'] = expert.get('congestion')
                    record[f'nothing_{i}'] = expert.get('nothing')
                    record[f'diagnosis_{i}'] = expert.get('diagnosis')
                    record[f'severity_{i}'] = expert.get('severity')

            records.append(record)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    df = pd.DataFrame(records)
    df.to_csv(output_file, index=False)
    print(f"\nSaved {len(df)} records to {output_file}")

    # Print summary
    print("\nStatus distribution:")
    print(df['status'].value_counts())

    print("\nCough detection score distribution:")
    print(df['cough_detected'].describe())

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../ml_data/coughvid/public_dataset")
    parser.add_argument("--output", default="../ml_data/coughvid/metadata_compiled.csv")
    args = parser.parse_args()

    compile_metadata(args.data_dir, args.output)
