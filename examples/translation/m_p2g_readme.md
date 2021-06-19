

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
CHECKPOINT_P2G=/content/drive/Shareddrives/mahouli249\\@gmail.com/exp/fairseq/wmt/p2g_test/

```