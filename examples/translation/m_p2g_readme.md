

```bash
# prepare
source /home/mhl/venv2/bin/activate
export PYTHONPATH=/content/drive/Shareddrives/mahouli249@gmail.com/git/fairseq:$PYTHONPATH

# Binarize the dataset
# lang - 后缀，pref - 前缀
DATA_ROOT=/content/drive/MyDrive/dataset/wmt/p2g_test
fairseq-preprocess  \
    --source-lang phone --target-lang text --trainpref $DATA_ROOT/org/valid.de --validpref $DATA_ROOT/org/valid.de --testpref $DATA_ROOT/org/test.de \
    --destdir $DATA_ROOT/data-bin --thresholdtgt 0 --thresholdsrc 0 \
    --workers 20

# train
export PYTHONPATH=/content/drive/Shareddrives/mahouli249@gmail.com/git/fairseq:$PYTHONPATH
EXP_ROOT=/content/drive/Shareddrives/mahouli249@gmail.com/exp/fairseq
fairseq-train \
    /content/drive/MyDrive/dataset/wmt/p2g_test/data-bin \
     --arch tutorial_simple_lstm \
    --encoder-dropout 0.2 --decoder-dropout 0.2 \
    --optimizer adam --lr 0.005 --lr-shrink 0.5 \
    --max-tokens 12000
    --save-dir $EXP_ROOT/test \
    --max-epoch 100 \
    --cpu \
    --tensorboard-logdir $EXP_ROOT/m_log/test | tee  $EXP_ROOT/m_log/test.log
```