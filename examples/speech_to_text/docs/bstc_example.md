
# BSTC
### 1.Data Preparation

#### prepare
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
ls /content/drive/MyDrive/exp/fairseq/bstc/asr/
# MyDrive
DriveRoot=/content/drive/MyDrive
export PYTHONPATH=/content/drive/Shareddrives/mahouli249@gmail.com/git/fairseq:$PYTHONPATH
BSTC_ROOT=$DriveRoot/dataset/bstc/root/
ASR_SAVE_DIR=$DriveRoot/exp/fairseq/bstc/asr
CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
ST_SAVE_DIR=$DriveRoot/exp/fairseq/bstc/st

# ShareDrive - transformer:
export PYTHONPATH=/content/drive/Shareddrives/mahouli249@gmail.com/git/fairseq:$PYTHONPATH
BSTC_ROOT=/content/drive/Shareddrives/mahouli249@gmail.com/dataset/bstc/root/
ASR_SAVE_DIR=/content/drive/Shareddrives/mahouli249@gmail.com/exp/fairseq/bstc/asr
ST_SAVE_DIR=/content/drive/Shareddrives/mahouli249@gmail.com/exp/fairseq/bstc/st
CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt

# ShareDrive - :
export PYTHONPATH=/content/drive/Shareddrives/mahouli249@gmail.com/git/fairseq:$PYTHONPATH
BSTC_ROOT=/content/drive/Shareddrives/mahouli249@gmail.com/dataset/bstc/root/
ASR_SAVE_DIR=/content/drive/Shareddrives/mahouli249@gmail.com/exp/fairseq/bstc2/asr
ST_SAVE_DIR=/content/drive/Shareddrives/mahouli249@gmail.com/exp/fairseq/bstc2/st
CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
```

#### preprocess
```bash
# preprocess :bstc style->covo style
python -m examples.speech_to_text.prep_bstc_data_1 -c ${BSTC_ROOT}

# ch asr
python -m debugpy --listen 0.0.0.0:5678 ./examples/speech_to_text/prep_bstc_data_2.py     --data-root ${BSTC_ROOT}      --src-vocab-type char  --src-vocab-size 3000     -s ch
python3 -m examples.speech_to_text.prep_bstc_data_2 \
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
## 注意dict.txt
## cd $BSTC_ROOT && rm dict.txt && cp spm_char3000_asr_ch.txt dict.txt
## load_checkpoint  - last_checkpoint
fairseq-train ${BSTC_ROOT} \
  --config-yaml config_asr_ch.yaml --train-subset train_asr_ch \
  --valid-subset dev_asr_ch --save-dir ${ASR_SAVE_DIR} \
  --num-workers 4 --max-tokens 40000 \
  --max-update 60000 --task speech_to_text \
  --criterion label_smoothed_cross_entropy --report-accuracy \
  --arch s2t_transformer_s --optimizer adam \
  --lr 2e-3 --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 --clip-norm 10.0 \
  --seed 1 --update-freq 8 \
  --max-epoch 140 \
  --tensorboard-logdir fairseq_asr_bstc_transformer_s_101_140 | tee fairseq_asr_bstc_transformer_s_101_140.log

  
  #s2t_transformer_s
#s2t_berard_256_3_3
```

#### Inference & Evaluation
```bash

python3 scripts/average_checkpoints.py \
  --inputs ${ASR_SAVE_DIR} --num-epoch-checkpoints 10 \
  --output "${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME}"

fairseq-generate ${BSTC_ROOT} \
  --config-yaml config_asr_ch.yaml --gen-subset dev_asr_ch --task speech_to_text \
  --path ${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME} --max-tokens 50000 --beam 5 \
  --scoring wer --wer-tokenizer 13a --wer-lowercase --wer-remove-punct \
  --tensorboard-logdir fairseq_asr_bstc_transformer_s_dev | tee fairseq_asr_bstc_transformer_s_dev.log

```

### 3.train
#### Train - ST
```bash
## 注意dict.txt
## cd $BSTC_ROOT && rm dict.txt && cp spm_unigram10000_st_ch_en.txt dict.txt

fairseq-train ${BSTC_ROOT} \
  --config-yaml config_st_ch_en.yaml --train-subset train_st_ch_en --valid-subset dev_st_ch_en \
  --save-dir ${ST_SAVE_DIR} --num-workers 4 --max-tokens 40000 --max-update 60000 \
  --task speech_to_text --criterion label_smoothed_cross_entropy --report-accuracy \
  --arch s2t_transformer_s --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 8 \
  --load-pretrained-encoder-from ${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME} \
  --max-epoch 100 \
  --tensorboard-logdir fairseq_st_bstc_transformer_s_51_100 | tee fairseq_st_bstc_transformer_s_51_100.log

```
#### Inference & Evaluation
```bash

python scripts/average_checkpoints.py \
  --inputs ${ST_SAVE_DIR} --num-epoch-checkpoints 10 \
  --output "${ST_SAVE_DIR}/${CHECKPOINT_FILENAME}"
fairseq-generate ${BSTC_ROOT} \
  --config-yaml config_st_ch_en.yaml --gen-subset dev_st_ch_en --task speech_to_text \
  --path ${ST_SAVE_DIR}/${CHECKPOINT_FILENAME} \
  --max-tokens 50000 --beam 5 --scoring sacrebleu \
  --tensorboard-logdir fairseq_st_bstc_transformer_s_100_dev | tee fairseq_st_bstc_transformer_s_100_dev.log
```

### 4.Interactive Decoding
```bash
/content/drive/MyDrive/dataset/bstc/root/train/cutting/104_0.wav
/content/drive/MyDrive/dataset/bstc/root/train/cutting/102534_229.wav
/content/drive/MyDrive/dataset/bstc/root/dev/cutting/6634_76.wav
/content/drive/MyDrive/dataset/my_test/ch/hdlwl_0008_00001.wav

# asr
fairseq-interactive ${BSTC_ROOT} --config-yaml config_asr_ch.yaml --task speech_to_text \
  --path ${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME} --max-tokens 50000 --beam 5


# st
fairseq-interactive ${BSTC_ROOT} --config-yaml config_st_ch_en.yaml \
  --task speech_to_text --path ${ST_SAVE_DIR}/${CHECKPOINT_FILENAME} \
  --max-tokens 50000 --beam 5 \
  --tensorboard-logdir fairseq_st_bstc_transformer_s_dev | tee fairseq_st_bstc_transformer_s_dev.log
```

