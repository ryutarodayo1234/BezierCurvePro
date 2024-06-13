from functools import partial
from pathlib import Path

import hydra
import joblib
import numpy as np
import torch
from hydra.utils import to_absolute_path
from nnmnkwii.io import hts
from omegaconf import DictConfig, OmegaConf
from scipy.io import wavfile
from tqdm import tqdm
from ttslearn.tacotron.gen import synthesis, synthesis_griffin_lim
from ttslearn.util import load_utt_list, optional_tqdm

import time


@hydra.main(config_path="conf/synthesis", config_name="config")
def my_app(config: DictConfig) -> None:
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(config.device)

    # acoustic model
    acoustic_config = OmegaConf.load(to_absolute_path(config.acoustic.model_yaml))
    acoustic_model = hydra.utils.instantiate(acoustic_config.netG).to(device)
    checkpoint = torch.load(
        to_absolute_path(config.acoustic.checkpoint),
        map_location=device,
    )
    acoustic_model.load_state_dict(checkpoint["state_dict"])
    acoustic_out_scaler = joblib.load(to_absolute_path(config.acoustic.out_scaler_path))
    acoustic_model.eval()

    # WaveNet
    # タイムスタンプ用の関数
    def log_time(message, start_time):
        elapsed_time = time.time() - start_time
        print(f"{message}: {elapsed_time:.2f} seconds")
        return time.time()

    start_time = time.time()

    # WaveNetの設定と初期化
    if config.use_wavenet:
        wavenet_config = OmegaConf.load(to_absolute_path(config.wavenet.model_yaml))
        wavenet_model = hydra.utils.instantiate(wavenet_config.netG).to(device)
        log_time("Model instantiation", start_time)
        
        checkpoint_start = time.time()
        checkpoint = torch.load(
            to_absolute_path(config.wavenet.checkpoint),
            map_location=device,
        )
        wavenet_model.load_state_dict(checkpoint["state_dict"])
        wavenet_model.eval()
        wavenet_model.remove_weight_norm_()
        log_time("Model loading and preparation", checkpoint_start)
    else:
        wavenet_model = None

    # ディレクトリの設定
    in_dir = Path(to_absolute_path(config.in_dir))
    out_dir = Path(to_absolute_path(config.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_time("Directory setup", start_time)

    # 音声ファイルリストの読み込み
    utt_ids = load_utt_list(to_absolute_path(config.utt_list))
    if config.reverse:
        utt_ids = utt_ids[::-1]
    lab_files = [in_dir / f"{utt_id.strip()}.lab" for utt_id in utt_ids]
    if config.num_eval_utts is not None and config.num_eval_utts > 0:
        lab_files = lab_files[:config.num_eval_utts]
    log_time("Loading utterance list", start_time)

    # 進捗バーの設定
    if config.tqdm == "tqdm":
        _tqdm = partial(tqdm, desc="wavenet generation", leave=False)
    else:
        _tqdm = lambda x: x  # No progress bar

    # 音声ファイルの処理
    for lab_file in _tqdm(lab_files):
        process_start = time.time()
        # 各音声ファイルの処理をここに記述
        # 例: 音声合成処理、結果の保存など
        log_time(f"Processing {lab_file.name}", process_start)

    log_time("Total execution time", start_time)

    # Run synthesis for each utt.
    for lab_file in optional_tqdm(config.tqdm, desc="Utterance")(lab_files):
        labels = hts.load(lab_file)
        if wavenet_model is None:
            wav = synthesis_griffin_lim(
                device, config.sample_rate, labels, acoustic_model, acoustic_out_scaler
            )
        else:
            wav = synthesis(
                device, config.sample_rate, labels, acoustic_model, wavenet_model, _tqdm
            )

        wav = np.clip(wav, -1.0, 1.0)

        utt_id = lab_file.stem
        out_wav_path = out_dir / f"{utt_id}.wav"
        wavfile.write(
            out_wav_path,
            rate=config.sample_rate,
            data=(wav * 32767.0).astype(np.int16),
        )


def entry():
    my_app()


if __name__ == "__main__":
    my_app()
