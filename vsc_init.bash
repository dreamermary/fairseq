# need run in terminal
# alias python=python3
# alias pip3=pip3
# export PYTHONPATH=/content/drive/Shareddrives/mahouli249@gmail.com/git/fairseq:$PYTHONPATH

# libri
LS_ROOT=/content/drive/MyDrive/dataset/libri/root
SAVE_DIR=/content/drive/MyDrive/fairseq/s2t/libri/checkpoint

# common voice
COVOST_ROOT=/content/drive/MyDrive/dataset/cv/root_v4_en/
COVOST_ROOT=/content/drive/MyDrive/dataset/cv/root_v4/


# bstc



# aleady run
pip3 uninstall numpy
# cd /content/drive/Shareddrives/mahouli249@gmail.com/git/fairseq && pip3 install --editable ./
pip3 install pandas torchaudio soundfile sentencepiece debugpy numpy omegaconf
apt install vim


# python3 -m debugpy --listen 0.0.0.0:5678 ./examples/speech_to_text/prep_bstc_data_2.py  --data-root ${COVOST_ROOT} --vocab-type char --src-lang en --tgt-lang zh-CN



