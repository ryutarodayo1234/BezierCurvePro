import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
from nnmnkwii.io import hts
from nnmnkwii.preprocessing import mulaw_quantize
from scipy.io import wavfile
from tqdm import tqdm

import os
import glob

import requests

url = 'https://raw.githubusercontent.com/ryutarodayo1234/BezierCurvePro/main/ttslearn/dsp.py'
filename = 'dsp.py'

response = requests.get(url)
if response.status_code == 200:
    with open(filename, 'wb') as file:
        file.write(response.content)

# ダウンロードしたモジュールをインポート
import dsp
from dsp import mulaw_quantize, logmelspectrogram

from ttslearn.tacotron.frontend.openjtalk import pp_symbols, text_to_sequence
from ttslearn.util import pad_1d


def get_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess for Tacotron",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("utt_list", type=str, help="utternace list")
    parser.add_argument("wav_root", type=str, help="wav root")
    parser.add_argument("lab_root", type=str, help="lab_root")
    parser.add_argument("out_dir", type=str, help="out directory")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of jobs")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Sample rate")
    parser.add_argument("--mu", type=int, default=256, help="mu")
    return parser

# pitch_to_number関数の定義
def pitch_to_number(pitch):
        # 各音名に対するピッチを定義
        pitch_map = {
            'C': 0,
            'C#': 1,
            'D-': 1,
            'D': 2,
            'D#': 3,
            'E-': 3,
            'E': 4,
            'F': 5,
            'F#': 6,
            'G-': 6,
            'G': 7,
            'G#': 8,
            'A-': 8,
            'A': 9,
            'A#': 10,
            'B-': 10,
            'B': 11
        }
        # ピッチを数値に変換して返す
        note, octave = pitch[:-1], int(pitch[-1])
        return pitch_map[note] + octave * 12

def preprocess(wav_file, lab_file, sr, mu, out_dir):
    # ラベルファイルを読み込む
    with open(lab_file, 'r') as f:
        labels = f.readlines()

    # 特徴量を格納するリスト
    features = []
    # 各行のラベル情報から特徴量を計算
    for line in labels:
        # 各行をタブで分割して情報を取得
        start_time, end_time, pitch = line.strip().split('\t')
        # 開始時間と終了時間をfloat型に変換
        start_time = float(start_time)
        end_time = float(end_time)
        # 音符の長さを計算
        duration = end_time - start_time
        # ピッチ（音高）を数値に変換
        pitch_value = pitch_to_number(pitch)
        # 特徴量として開始時間、終了時間、ピッチ、音符の長さを追加
        features.append([start_time, end_time, pitch_value, duration])

    # 特徴量のリストをNumPy配列に変換
    in_feats = np.array(features, dtype=np.float32)

    # 保存するファイル名を指定
    file_name = os.path.basename(wav_file).replace(".wav", "-feats.npy")
    out_file = os.path.join(out_dir, file_name)

    # 特徴量の保存
    np.save(out_file, in_feats, allow_pickle=False)

if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])

    with open(args.utt_list) as f:
        utt_ids = [utt_id.strip() for utt_id in f]
    wav_files = [Path(args.wav_root) / f"{utt_id}.wav" for utt_id in utt_ids]
    lab_files = [Path(args.lab_root) / f"{utt_id}.lab" for utt_id in utt_ids]

    in_dir = Path(args.out_dir) / "in_tacotron"
    out_dir = Path(args.out_dir) / "out_tacotron"
    wave_dir = Path(args.out_dir) / "out_wavenet"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    wave_dir.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(args.n_jobs) as executor:
        futures = [
            executor.submit(
                preprocess,
                wav_file,
                lab_file,
                args.sample_rate,
                args.mu,
                in_dir,
                out_dir,
                wave_dir,
            )
            for wav_file, lab_file in zip(wav_files, lab_files)
        ]
        for future in tqdm(futures):
            future.result()
