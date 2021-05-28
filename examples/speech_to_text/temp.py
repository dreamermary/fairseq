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

# filename = os.path.join('/content/drive/MyDrive/dataset/bstc/root','dev_st_ch_en.tsv')
# # filename = os.path.join('/content/drive/MyDrive','/dataset/bstc/root','train_st_ch_en.tsv')

# # train_txt=[]

# # with open(filename) as f:
# #     ls = f.readlines()
# #     for l in ls:
# #         train_txt.append((str(l)).split('\t')[3])


    
# train_txt = [1,2,3,4]

# print(train_txt[1:]) 

# 批量修改文件名 - （复制）----------------
# root = '/content/drive/Shareddrives/mahouli249@gmail.com/share/bstc/root'
# filenames = os.listdir(root)
# for name in filenames:
#     os.rename(os.path.join(root,name),os.path.join(root,name[0:-4]) )
#     # print(os.path.join(root,name[0:-4]))
# # print(filenames)


# filename1 = '/content/drive/Shareddrives/mahouli249@gmail.com/dataset/bstc/root/train_asr_ch.tsv'
# filename2 = '/content/drive/Shareddrives/mahouli249@gmail.com/dataset/bstc/root/train_st_ch_en.tsv'
# count1 = len(open(filename1,'rU').readlines())
# count2 = len(open(filename2,'rU').readlines())

# count1 ="99"
# count2 ="88"

# print(count1)
# print(count2)


# gene_vocab
#----
from pathlib import Path
from tempfile import NamedTemporaryFile
from examples.speech_to_text.data_utils import (
    create_zip,
    extract_fbank_features,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    save_df_to_tsv,
)
train_text = []
root = Path('/content/drive/Shareddrives/mahouli249@gmail.com/dataset/bstc/root').absolute()
filename = root /'train_asr_ch.tsv'
outname = root / 'test_vocab'
# if args.tgt_lang is not None:
#     filename = root /'train_st_ch_en.tsv'

# 读取
with open(filename.as_posix()) as f:
    ls = f.readlines()
    for l in ls:
        train_text.append((str(l)).split('\t')[3])
train_text = train_text[1:]

# 临时写入
with NamedTemporaryFile(mode="w") as f:
    for t in train_text:
        f.write(t + "\n")
    gen_vocab(
        Path(f.name),
        outname,
        'char',
        '3000',
    )


