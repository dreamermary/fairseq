## libri
```bash
LS_ROOT=/content/drive/MyDrive/dataset/libri/root
SAVE_DIR=/content/drive/MyDrive/fairseq/s2t/libri/checkpoint
python -m examples.speech_to_text.prep_librispeech_data \
  --output-root ${LS_ROOT} --vocab-type unigram --vocab-size 10000
fairseq-train ${LS_ROOT} --save-dir ${SAVE_DIR} \
  --config-yaml config.yaml --train-subset train-clean-100 --valid-subset test-clean \
  --num-workers 4 --max-tokens 40000 --max-update 300000 \
  --task speech_to_text --criterion label_smoothed_cross_entropy --report-accuracy \
  --arch s2t_transformer_s --share-decoder-input-output-embed \
  --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 10000 \
  --clip-norm 10.0 --seed 1 --update-freq 8 \
  --max-epoch 10

import os
print('clumba:-----:%s：%s'%(str(os.path.basename(__file__)).split('.')[0], str())) ##
```

## common voice
```bash
COVOST_ROOT=/content/drive/MyDrive/dataset/cv/root_v4_en/

```


## bstc
python3 -m examples.speech_to_text.prep_bstc_data_1 -c /content/drive/MyDrive/dataset/bstc/root


translation:
  path
  translation
  split