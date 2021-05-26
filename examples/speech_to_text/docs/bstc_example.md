
# BSTC
## Data Preparation

### 1. prepare
```bash
一键回到解放前 
# config
git config --global user.name "dreamermary"
git config --global user.email 1341584939@qq.com
git remote add origin https://github.com/dreamermary/fairseq.git

# prepare
pip uninstall -y numpy
pip install pandas torchaudio soundfile sentencepiece debugpy numpy omegaconf --editable ./
apt install vim screen
```
```bash
# MyDrive:
DriveRoot=/content/drive/MyDrive
export PYTHONPATH=/content/drive/Shareddrives/mahouli249@gmail.com/git/fairseq:$PYTHONPATH
BSTC_ROOT=$DriveRoot/dataset/bstc/root/
ASR_SAVE_DIR=$DriveRoot/exp/fairseq/bstc/asr
CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
ST_SAVE_DIR=$DriveRoot/exp/fairseq/bstc/st

# ShareDrive:
export PYTHONPATH=/content/drive/Shareddrives/mahouli249@gmail.com/git/fairseq:$PYTHONPATH
BSTC_ROOT=/content/drive/Shareddrives/mahouli249@gmail.com/dataset/bstc/root/
ASR_SAVE_DIR=/content/drive/Shareddrives/mahouli249@gmail.com/exp/fairseq/bstc/asr
CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
ST_SAVE_DIR=/content/drive/Shareddrives/mahouli249@gmail.com/exp/fairseq/bstc/st
```

### 2. preprocess
```bash
# preprocess :bstc style->covo style
python -m examples.speech_to_text.prep_bstc_data_1 -c ${BSTC_ROOT}

# ch asr
python -m examples.speech_to_text.prep_bstc_data_2 \
    --data-root ${BSTC_ROOT}  \
    --src-vocab-type char  --src-vocab-size 3000 \
    -s ch
# ch-en st
python -m examples.speech_to_text.prep_bstc_data_2 \
    --data-root ${BSTC_ROOT}  \
    --trg-vocab-type unigram  --trg-vocab-size 10000 \
    -s ch -t en


# ps:
#code path
/content/drive/Shareddrives/mahouli249@gmail.com/git/fairseq
#dataset path 

```

### 3.train
#### Train - ASR
```bash
fairseq-train ${BSTC_ROOT} \
  --config-yaml config_asr_ch.yaml --train-subset train_asr_ch \
  --valid-subset dev_asr_ch --save-dir ${ASR_SAVE_DIR} 
  --num-workers 4 --max-tokens 40000 \
  --max-update 60000 --task speech_to_text \
  --criterion label_smoothed_cross_entropy --report-accuracy \
  --arch s2t_transformer_s --optimizer adam \
  --lr 2e-3 --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 --clip-norm 10.0 \
  --seed 1 --update-freq 8 \
  --max-epoch 10

```

#### Inference & Evaluation
```bash

python scripts/average_checkpoints.py \
  --inputs ${ASR_SAVE_DIR} --num-epoch-checkpoints 10 \
  --output "${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME}"

fairseq-generate ${BSTC_ROOT} \
  --config-yaml config_asr_ch.yaml --gen-subset dev_asr_ch --task speech_to_text \
  --path ${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME} --max-tokens 50000 --beam 5 \
  --scoring wer --wer-tokenizer 13a --wer-lowercase --wer-remove-punct


```

### 3.train
#### Train - ST
```bash

fairseq-train ${BSTC_ROOT} \
  --config-yaml config_st_ch_en.yaml --train-subset train_st_ch_en --valid-subset dev_st_ch_en \
  --save-dir ${ST_SAVE_DIR} --num-workers 4 --max-tokens 40000 --max-update 60000 \
  --task speech_to_text --criterion label_smoothed_cross_entropy --report-accuracy \
  --arch s2t_transformer_s --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8 \
  --load-pretrained-encoder-from ${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME} \
  --max-epoch 10

```
#### Inference & Evaluation
```bash

python scripts/average_checkpoints.py \
  --inputs ${ST_SAVE_DIR} --num-epoch-checkpoints 10 \
  --output "${ST_SAVE_DIR}/${CHECKPOINT_FILENAME}"
fairseq-generate ${BSTC_ROOT} \
  --config-yaml config_st_ch_en.yaml --gen-subset dev_st_ch_en --task speech_to_text \
  --path ${ST_SAVE_DIR}/${CHECKPOINT_FILENAME} \
  --max-tokens 50000 --beam 5 --scoring sacrebleu
```

### 4.Interactive Decoding
```bash
fairseq-interactive ${BSTC_ROOT} --config-yaml config_st_ch_en.yaml \
  --task speech_to_text --path ${SAVE_DIR}/${CHECKPOINT_FILENAME} \
  --max-tokens 50000 --beam 5
```

