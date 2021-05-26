# import re
# import torchaudio
# EN_STOP_CHAR = '[\.。,，、/\?？:：“”\"‘’\'\{\}\[\]|\\<>\|《》~`·！!@#\$%\^&\*\(\)￥……（）—=_-【】]'

# EN_STOP_CHAR="[+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）:;：；‘’“”《》<>\?]"
# ZH_STOP_CHAR="[+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）:;：；‘’“”《》<>ZXCVBNMASDFGHJKLQWERTYUIOPzxcvbnmasdfghjklqwertyuiop\?]"
# src = "This boy is a nuisance, you might say."
# trg = "我吃lL，饭了?。"
# src = re.sub(EN_STOP_CHAR, "", src)
# trg = re.sub(ZH_STOP_CHAR, "", trg)

# from string import punctuation
# punc = punctuation + u'.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：'

# print(src)
# print(trg)

# path = r'/content/drive/Shareddrives/mahouli249@gmail.com/dataset/bstc/root/train/cutting/4513_159.wav'
# waveform, sample_rate = torchaudio.load(path)
# print(waveform) 
# print(sample_rate)

# print(str.lower('WWGGggy yY Y'))
# 111

#--------------------------------------------
# import os

# DriveRoot=/content/drive/MyDrive
# BSTC_ROOT=$DriveRoot/dataset/bstc/root/
# ASR_SAVE_DIR=$DriveRoot/exp/fairseq/bstc/asr
# ST_SAVE_DIR=$DriveRoot/exp/fairseq/bstc/st

# os.makedirs("/home/jb51/data")

# root = Path(args.data_root).absolute() 
# if not root.is_dir():
# feature_root = root / "fbank80"
# feature_root.mkdir(exist_ok=True)

#--------------------------------------------

import csv
import os

filename = os.path.join('/content/drive/MyDrive/dataset/bstc/root','dev_st_ch_en.tsv')
# filename = os.path.join('/content/drive/MyDrive','/dataset/bstc/root','train_st_ch_en.tsv')

# train_txt=[]

# with open(filename) as f:
#     ls = f.readlines()
#     for l in ls:
#         train_txt.append((str(l)).split('\t')[3])


    
train_txt = [1,2,3,4]

print(train_txt[1:]) 
