import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

def get_parser():
    parser = argparse.ArgumentParser(description="Fit scalers")
    parser.add_argument("utt_list", type=str, help="utternace list")
    parser.add_argument("in_dir", type=str, help="in directory")
    parser.add_argument("out_path", type=str, help="Output path")
    parser.add_argument("--external_scaler", type=str, help="External scaler")
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])
    in_dir = Path(args.in_dir)
    if args.external_scaler is not None:
        scaler = joblib.load(args.external_scaler)
    else:
        scaler = MinMaxScaler()  # MinMaxScalerを使用する
    normalized_data_list = []
    with open(args.utt_list) as f:
        for utt_id in tqdm(f):
            c = np.load(in_dir / f"{utt_id.strip()}-feats.npy")
            normalized_data = scaler.fit_transform(c)  # データを正規化
            normalized_data_list.append(normalized_data)
        # リストをnumpyの多次元配列に変換
        normalized_data_array = np.array(normalized_data_list)
        np.save(args.out_path, normalized_data_array)
