import argparse
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

def get_parser():
    parser = argparse.ArgumentParser(description="Normalize features")
    parser.add_argument("utt_list", type=str, help="Utterance list")
    parser.add_argument("in_dir", type=str, help="Input directory")
    parser.add_argument("out_path", type=str, help="Output path for scaler")
    return parser

def normalize_features(data):
    # データの正規化処理を実装
    # ここでは例として、最小値と最大値を使用した正規化を行います
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])
    in_dir = Path(args.in_dir)
    
    normalized_data_list = []

    with open(args.utt_list) as f:
        for utt_id in tqdm(f):
            c = np.load(in_dir / f"{utt_id.strip()}-feats.npy")
            normalized_data = normalize_features(c)
            normalized_data_list.append(normalized_data)

    # 正規化したデータの統計情報を保存する代わりに、正規化されたデータ自体を保存します
    np.save(args.out_path, np.array(normalized_data_list))
