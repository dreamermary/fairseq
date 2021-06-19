 os.makedirs(args.destdir, exist_ok=True)
```bash
# prepare
source /home/mhl/venv2/bin/activate
export PYTHONPATH=/content/drive/Shareddrives/mahouli249@gmail.com/git/fairseq:$PYTHONPATH
```    

## Phone
 ```bash
Phone - valid - test - en_de

# learn & apply bpe in phone dataset
bash m_prepare-phone.sh

# dict & Binarize 
TEXT=/home/mhl/dataset/wmt2/bpephone # 数据集位置
fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref $TEXT/bpe.phone.train \
    --validpref $TEXT/bpe.phone.valid \
    --testpref $TEXT/bpe.phone.test \
    --destdir data-bin/phone_com_en_de \
    --thresholdtgt 0 --thresholdsrc 0 \
    --workers 20


# train
CHECKPOINT_PHONE=/content/drive/Shareddrives/mahouli249\\@gmail.com/exp/fairseq/wmt/phone/
CHECKPOINT_PHONE=/content/drive/Shareddrives/mahouli249@gmail.com/exp/fairseq/wmt/CHECKPOINT_PHONE
fairseq-train \
    /content/drive/MyDrive/dataset/wmt/phone/data-bin/phone_com_en_de/ \
    --arch fconv_wmt_en_de \
    --dropout 0.2 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer nag --clip-norm 0.1 \
    --lr 0.5 --lr-scheduler fixed --force-anneal 50 \
    --max-tokens 4000 \
    --save-dir $CHECKPOINT_PHONE \
    --max-epoch 100 \
    --tensorboard-logdir m_log/phone_com_en_de_5 | tee  m_log/phone_com_en_de_5.log

# Evaluate
fairseq-generate /content/drive/MyDrive/dataset/wmt/phone/data-bin/phone_com_en_de/ \
    --path CHECKPOINT_PHONE/checkpoint_best.pt \
    --beam 5 --remove-bpe \
    --tensorboard-logdir m_log/gene_phone_com_en_de_21 | tee  m_log/gene_phone_com_en_de_21.log

 ```
 ---

## Text
 ```bash
text - valid - test - en_de
data-bin/text_test_en_de
checkpoints/text_test_en_de

# learn & apply bpe in phone dataset
bash m_prepare-phone.sh

# dict & Binarize 
TEXT=/home/mhl/git/clone/fairseq/examples/translation/wmt17_en_de # 数据集位置
CUDA_VISIBLE_DEVICES=14,15 fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref $TEXT/train \
    --validpref $TEXT/valid \
    --testpref $TEXT/test \
    --destdir data-bin/text_com_en_de \
    --thresholdtgt 0 --thresholdsrc 0 \
    --workers 20


# train

CHECKPOINT_TEXT=/content/drive/MyDrive/exp/fairseq/wmt/text/
mkdir -p $CHECKPOINT_TEXT
fairseq-train \
    /content/drive/MyDrive/dataset/wmt/text/data-bin/text_com_en_de \
    --arch fconv_wmt_en_de \
    --dropout 0.2 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer nag --clip-norm 0.1 \
    --lr 0.5 --lr-scheduler fixed --force-anneal 50 \
    --max-tokens 4000 \
    --save-dir  $CHECKPOINT_TEXT \
    --max-epoch 100 \
    --tensorboard-logdir m_log/text_com_en_de_1_100 | tee m_log/text_com_en_de_1_100.log

# Evaluate
fairseq-generate data-bin/text_test_en_de \
    --path checkpoints/phone_test2_en_de/checkpoint_best.pt \
    --beam 5 --remove-bpe \
     --tensorboard-logdir m_log/gene_text_com_en_de_1_100 | tee m_log/gene_text_com_en_de_1_100.log

 ```