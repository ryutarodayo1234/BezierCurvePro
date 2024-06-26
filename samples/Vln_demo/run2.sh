#!/bin/bash

set -e
set -u
set -o pipefail

function xrun () {
    set -x
    $@
    set +x
}

script_dir=$(cd $(dirname ${BASH_SOURCE:-$0}); pwd)
COMMON_ROOT=../common
. $COMMON_ROOT/yaml_parser.sh || exit 1;

eval $(parse_yaml "./config.yaml" "")

train_set="train"
dev_set="dev"
eval_set="eval"
datasets=($train_set $dev_set $eval_set)
testsets=($eval_set)

stage=0
stop_stage=0

. $COMMON_ROOT/parse_options.sh || exit 1;

dumpdir=dump
dump_org_dir=$dumpdir/${spk}_sr${sample_rate}/org
dump_norm_dir=$dumpdir/${spk}_sr${sample_rate}/norm

# exp name
if [ -z ${tag:=} ]; then
    expname=${spk}_sr${sample_rate}
else
    expname=${spk}_sr${sample_rate}_${tag}
fi
expdir=exp/$expname

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data download"
    mkdir -p downloads
    if [ ! -d downloads/corpus_files/file1 ]; then
        cd downloads/corpus_files
        curl -LO https://github.com/ryutarodayo1234/BezierCurvePro/raw/main/samples/Vln_demo/corpus_files.zip
        unzip -o corpus_files
        cd -
    fi
    if [ ! -d downloads/lab_files/file1 ]; then
        cd downloads/lab_files
        curl -LO https://github.com/ryutarodayo1234/BezierCurvePro/raw/main/samples/Vln_demo/lab_files.zip
        unzip -o lab_files.zip
        ln -s jlab_files
        cd -
    fi
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    echo "train/dev/eval split"
    mkdir -p data
    find $wav_root -name "*.wav" -exec basename {} .wav \; | sort > data/utt_list.txt
    #utt_list.txt ファイルの上位 4700 行を data/train.list に書き込みます。これは訓練データに使用されます。
    #utt_list.txt ファイルの下位 300 行を data/deveval.list に書き込みます。
    #deveval.list ファイルの上位 200 行を data/dev.list に書き込みます。これは開発データに使用されます。
    #deveval.list ファイルの下位 100 行を data/eval.list に書き込みます。これは評価データに使用されます。
    #不要な deveval.list ファイルを削除します。
    head -n 15 data/utt_list.txt > data/train.list
    tail -n 1 data/utt_list.txt > data/deveval.list
    head -n 5 data/deveval.list > data/dev.list
    tail -n 5 data/deveval.list > data/eval.list
    rm -f data/deveval.list
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Feature generation for Tacotron"
    for s in ${datasets[@]}; do
        xrun python preprocess.py data/$s.list $wav_root $lab_root \
            $dump_org_dir/$s --n_jobs $n_jobs \
            --sample_rate $sample_rate --mu $mu
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: feature normalization"
    for typ in "tacotron"; do
       for inout in "out"; do
            xrun python $COMMON_ROOT/fit_scaler.py data/train.list \
                $dump_org_dir/$train_set/${inout}_${typ} \
                $dump_org_dir/${inout}_${typ}_scaler.joblib
        done
    done

    mkdir -p $dump_norm_dir
    cp -v $dump_org_dir/*.joblib $dump_norm_dir/

    for s in ${datasets[@]}; do
        for typ in "tacotron"; do
            for inout in "out" "in"; do
                if [ $inout == "in" ]; then
                    cp -r $dump_org_dir/$s/${inout}_${typ} $dump_norm_dir/$s/
                    continue
                fi
                xrun python $COMMON_ROOT/preprocess_normalize.py data/$s.list \
                    $dump_org_dir/${inout}_${typ}_scaler.joblib \
                    $dump_org_dir/$s/${inout}_${typ}/ \
                    $dump_norm_dir/$s/${inout}_${typ}/ --n_jobs $n_jobs
            done
        done
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Training Tacotron"
    xrun python train_tacotron.py model=$acoustic_model tqdm=$tqdm \
        data.train.utt_list=data/train.list \
        data.train.in_dir=$dump_norm_dir/$train_set/in_tacotron/ \
        data.train.out_dir=$dump_norm_dir/$train_set/out_tacotron/ \
        data.dev.utt_list=data/dev.list \
        data.dev.in_dir=$dump_norm_dir/$dev_set/in_tacotron/ \
        data.dev.out_dir=$dump_norm_dir/$dev_set/out_tacotron/ \
        train.out_dir=$expdir/${acoustic_model} \
        train.log_dir=tensorboard/${expname}_${acoustic_model} \
        train.max_train_steps=$tacotron_train_max_train_steps \
        data.batch_size=$tacotron_data_batch_size \
        cudnn.benchmark=$cudnn_benchmark cudnn.deterministic=$cudnn_deterministic
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Training WaveNet vocoder"
    xrun python train_wavenet.py model=$wavenet_model tqdm=$tqdm \
        data.train.utt_list=data/train.list \
        data.train.in_dir=$dump_norm_dir/$train_set/out_tacotron/ \
        data.train.out_dir=$dump_org_dir/$train_set/out_wavenet/ \
        data.dev.utt_list=data/dev.list \
        data.dev.in_dir=$dump_norm_dir/$dev_set/out_tacotron/ \
        data.dev.out_dir=$dump_org_dir/$dev_set/out_wavenet/ \
        train.out_dir=$expdir/${wavenet_model} \
        train.log_dir=tensorboard/${expname}_${wavenet_model} \
        train.max_train_steps=$wavenet_train_max_train_steps \
        data.batch_size=$wavenet_data_batch_size \
        cudnn.benchmark=$cudnn_benchmark cudnn.deterministic=$cudnn_deterministic
fi

'''

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Training Tacotron"
    if [ ${finetuning} = "true" ] && [ -z ${pretrained_acoustic_checkpoint} ]; then
        pretrained_acoustic_checkpoint=$PWD/../../jsut/tacotron2_pwg/exp/jsut_sr${sample_rate}/${acoustic_model}/${acoustic_eval_checkpoint}
        if [ ! -e $pretrained_acoustic_checkpoint ]; then
            echo "Please first train a acoustic model for JSUT corpus!"
            echo "Expected model path: $pretrained_acoustic_checkpoint"
            exit 1
        fi
    fi
    xrun python train_tacotron.py model=$acoustic_model tqdm=$tqdm \
        data.train.utt_list=data/train.list \
        data.train.in_dir=$dump_norm_dir/$train_set/in_tacotron/ \
        data.train.out_dir=$dump_norm_dir/$train_set/out_tacotron/ \
        data.dev.utt_list=data/dev.list \
        data.dev.in_dir=$dump_norm_dir/$dev_set/in_tacotron/ \
        data.dev.out_dir=$dump_norm_dir/$dev_set/out_tacotron/ \
        train.out_dir=$expdir/${acoustic_model} \
        train.log_dir=tensorboard/${expname}_${acoustic_model} \
        train.max_train_steps=$tacotron_train_max_train_steps \
        data.batch_size=$tacotron_data_batch_size \
        cudnn.benchmark=$cudnn_benchmark cudnn.deterministic=$cudnn_deterministic #\
        #train.pretrained.checkpoint=$pretrained_acoustic_checkpoint
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Training Parallel WaveGAN"
    if [ ${finetuning} = "true" ] && [ -z ${pretrained_vocoder_checkpoint} ]; then
        voc_expdir=$PWD/../../jsut/tacotron2_pwg/exp/jsut_sr${sample_rate}/${vocoder_model}
        pretrained_vocoder_checkpoint="$(ls -dt "$voc_expdir"/*.pkl | head -1 || true)"
        if [ ! -e $pretrained_vocoder_checkpoint ]; then
            echo "Please first train a PWG model for JSUT corpus!"
            echo "Expected model path: $pretrained_vocoder_checkpoint"
            exit 1
        fi
        extra_args="--resume $pretrained_vocoder_checkpoint"
    else
        extra_args=""
    fi
    xrun parallel-wavegan-train --config $parallel_wavegan_config \
        --train-dumpdir $dump_norm_dir/$train_set/out_tacotron \
        --dev-dumpdir $dump_norm_dir/$dev_set/out_tacotron/ \
        --outdir $expdir/$vocoder_model $extra_args
fi

'''

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Synthesis waveforms by the griffin-lim algorithm"
    for s in ${testsets[@]}; do
        xrun python synthesis.py utt_list=./data/$s.list tqdm=$tqdm \
            in_dir=${lab_root} \
            out_dir=$expdir/synthesis_${acoustic_model}_griffin_lim/$s \
            sample_rate=$sample_rate \
            acoustic.checkpoint=$expdir/${acoustic_model}/$acoustic_eval_checkpoint \
            acoustic.out_scaler_path=$dump_norm_dir/out_tacotron_scaler.joblib \
            acoustic.model_yaml=$expdir/${acoustic_model}/model.yaml \
            reverse=$reverse num_eval_utts=$num_eval_utts
    done
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: Synthesis waveforms by WaveNet vocoder"
    for s in ${testsets[@]}; do
        xrun python synthesis.py utt_list=./data/$s.list tqdm=$tqdm \
            in_dir=${lab_root} \
            out_dir=$expdir/synthesis_${acoustic_model}_${wavenet_model}/$s \
            sample_rate=$sample_rate \
            acoustic.checkpoint=$expdir/${acoustic_model}/$acoustic_eval_checkpoint \
            acoustic.out_scaler_path=$dump_norm_dir/out_tacotron_scaler.joblib \
            acoustic.model_yaml=$expdir/${acoustic_model}/model.yaml \
            wavenet.checkpoint=$expdir/${wavenet_model}/$wavenet_eval_checkpoint \
            wavenet.model_yaml=$expdir/${wavenet_model}/model.yaml \
            use_wavenet=true reverse=$reverse num_eval_utts=$num_eval_utts
    done
fi

if [ ${stage} -le 98 ] && [ ${stop_stage} -ge 98 ]; then
    echo "Create tar.gz to share experiments"
    rm -rf tmp/exp
    mkdir -p tmp/exp/$expname
    for model in $acoustic_model $wavenet_model; do
        rsync -avr $expdir/$model tmp/exp/$expname/ --exclude "epoch*.pth"
    done
    rsync -avr $expdir/synthesis_${acoustic_model}_griffin_lim tmp/exp/$expname/ --exclude "epoch*.pth"
    rsync -avr $expdir/synthesis_${acoustic_model}_${wavenet_model} tmp/exp/$expname/ --exclude "epoch*.pth"
    cd tmp
    tar czvf tacotron_exp.tar.gz exp/
    mv tacotron_exp.tar.gz ..
    cd -
    rm -rf tmp
    echo "Please check tacotron_exp.tar.gz"
fi

if [ ${stage} -le 99 ] && [ ${stop_stage} -ge 99 ]; then
    echo "Pack models for TTS"
    dst_dir=tts_models/${expname}_${acoustic_model}_${wavenet_model}
    mkdir -p $dst_dir

    # global config
    cat > ${dst_dir}/config.yaml <<EOL
sample_rate: ${sample_rate}
mu: ${mu}
acoustic_model: ${acoustic_model}
wavenet_model: ${wavenet_model}
EOL

    # Stats
    python $COMMON_ROOT/scaler_joblib2npy.py $dump_norm_dir/out_tacotron_scaler.joblib $dst_dir

    # Acoustic model
    python $COMMON_ROOT/clean_checkpoint_state.py $expdir/${acoustic_model}/$acoustic_eval_checkpoint \
        $dst_dir/acoustic_model.pth
    cp $expdir/${acoustic_model}/model.yaml $dst_dir/acoustic_model.yaml

    # WaveNet
    python $COMMON_ROOT/clean_checkpoint_state.py $expdir/${wavenet_model}/$wavenet_eval_checkpoint \
        $dst_dir/wavenet_model.pth
    cp $expdir/${wavenet_model}/model.yaml $dst_dir/wavenet_model.yaml

    echo "All the files are ready for TTS!"
    echo "Please check the $dst_dir directory"
 fi