import argparse
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch

def get_parser():
    parser = argparse.ArgumentParser(description="特徴量の正規化")
    parser.add_argument("utt_list", type=str, help="発話リスト")
    parser.add_argument("in_dir", type=str, help="入力ディレクトリ")
    parser.add_argument("out_path", type=str, help="出力パス")
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])
    in_dir = Path(args.in_dir)
    normalized_data_list = []

    with open(args.utt_list) as f:
        for utt_id in tqdm(f):
            c = np.load(in_dir / f"{utt_id.strip()}-feats.npy")
            print("cの形状:", c.shape)

            # 最小値と最大値を取得
            min_vals = np.min(c, axis=0)
            max_vals = np.max(c, axis=0)

            # 正規化
            normalized_data = (c - min_vals) / (max_vals - min_vals)
            print("normalized_dataの形状:", normalized_data.shape)

            # PyTorchのTensorに変換
            normalized_data = torch.from_numpy(normalized_data)

            normalized_data_list.append(normalized_data)
            print("normalized_data_listの長さ:", len(normalized_data_list))

    # 可変長さのデータをパディング
    padded_sequences = torch.nn.utils.rnn.pad_sequence(normalized_data_list, batch_first=True)

    # PyTorchのTensorをNumPy配列に変換して保存
    np.save(args.out_path, padded_sequences.numpy())