
root=/content/drive/Shareddrives/mahouli249@gmail.com/dataset/wmt/en_de
TRAIN=$root/phone/phone.data.de # 所有单语phone数据集
BPEROOT=subword-nmt/subword_nmt
BPE_CODE=$root/code.de
BPE_TOKENS=40000
tgt=de
iroot=$root/org
oroot=$root/bpephone

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $tgt; do
    for f in train.$L test.$L valid.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $iroot/$f > $oroot/bpe.$f
    done
done
