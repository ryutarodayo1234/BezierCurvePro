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
from ttslearn.tacotron.frontend.openjtalk import pp_symbols, text_to_sequence


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
    parser.add_argument("wav_root", type=str, help="wav_root")
    parser.add_argument("lab_root", type=str, help="lab_root")
    parser.add_argument("out_dir", type=str, help="out directory")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of jobs")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Sample rate")
    parser.add_argument("--mu", type=int, default=256, help="mu")
    return parser


def preprocess(
    wav_file,
    lab_file,
    sr,
    mu,
    in_dir,
    out_dir,
    wave_dir,
):
    # デバッグ用ログ
    print("Starting preprocess for:", os.path.basename(lab_file))

    # ラベルファイルのルートディレクトリ
    lab_root = "downloads/lab_files"

    # wavファイルのルートディレクトリ
    wav_root = "downloads/corpus_files"

    # ラベルファイルのリストを取得
    label_files = glob.glob(os.path.join(lab_root, "files*"))

    for lab_file in label_files:
        # wavファイルのパスを取得
        wav_file = os.path.join(wav_root, os.path.basename(lab_file).replace(".lab", ".wav"))

        # ラベルファイル名とwavファイル名が一致することを確認
        assert os.path.splitext(wav_file)[0] == os.path.splitext(lab_file)[0]

        # ラベルファイルを読み込む
        with open(lab_file, 'r') as f:
            labels = f.read()

        # デバッグ用ログ
        print("Labels:", labels)

        # 韻律記号付き音素列の抽出
        PP = pp_symbols(labels)
        in_feats = np.array(text_to_sequence(PP), dtype=np.int64)

        # wavファイルを読み込む
        _sr, x = wavfile.read(wav_file)

        # メルスペクトログラムの計算
        if x.dtype in [np.int16, np.int32]:
            x = (x / np.iinfo(x.dtype).max).astype(np.float64)
        x = librosa.resample(y=x, orig_sr=_sr, target_sr=sr)
        out_feats = logmelspectrogram(x, sr)

        # デバッグ用ログ
        print("Out feats shape:", out_feats.shape)

        """
        # 冒頭と末尾の非音声区間の長さを調整
        assert "sil" in labels.contexts[0] and "sil" in labels.contexts[-1]
        start_frame = int(labels.start_times[1] / 125000)
        end_frame = int(labels.end_times[-2] / 125000)

        # 冒頭： 50 ミリ秒、末尾： 100 ミリ秒
        start_frame = max(0, start_frame - int(0.050 / 0.0125))
        end_frame = min(len(out_feats), end_frame + int(0.100 / 0.0125))

        out_feats = out_feats[start_frame:end_frame]

        # 時間領域で音声の長さを調整
        x = x[int(start_frame * 0.0125 * sr) :]
        length = int(sr * 0.0125) * out_feats.shape[0]
        x = pad_1d(x, length) if len(x) < length else x[:length]
        """
        
        # 特徴量のアップサンプリングを行う都合上、音声波形の長さはフレームシフトで割り切れる必要があります
        assert len(x) % int(sr * 0.0125) == 0

        # mu-law量子化
        x = mulaw_quantize(x, mu)

        # save to files
        utt_id = lab_file.stem

        # デバッグ用ログ
        print("Saving files for:", utt_id)

        np.save(in_dir / f"{utt_id}-feats.npy", in_feats, allow_pickle=False)
        np.save(
            out_dir / f"{utt_id}-feats.npy",
            out_feats.astype(np.float32),
            allow_pickle=False,
        )
        np.save(
            wave_dir / f"{utt_id}-feats.npy",
            x.astype(np.int64),
            allow_pickle=False,
        )
        # デバッグ用ログ
        print("Preprocessing completed.")



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