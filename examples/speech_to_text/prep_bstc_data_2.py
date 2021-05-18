# -*- coding: utf-8 -*-
# @Author  :   clumba
# @File    :   cutting_bstc.py
# @Time    :   2021/4/21 10:26

import argparse
import logging
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from typing import Optional, Tuple

import pandas as pd
import torchaudio
from examples.speech_to_text.data_utils import (
    create_zip,
    extract_fbank_features,
    filter_manifest_df,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    load_df_from_tsv,
    save_df_to_tsv,
)
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio.datasets.utils import download_url, extract_archive
from tqdm import tqdm
from pathlib import PurePath
import json

SPLITS = ["train","dev"]
MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]

class BSTC(Dataset):
    def __init__(self,root,split):
        self.data = []
        self.root = root
        path = self.root / split / (split+'.tsv')
        # assert path.is_file()
        self.data = open(path,'r',encoding='utf-8').readlines()
        self.root = root
        self.split = split

    def __getitem__(
        self, n: int
    ) -> Tuple[Tensor, int, str, str, Optional[str], str, str]:
        data = self.data[n]
        # data = json.loads(data.strip())
        data = eval(data)
        path = self.root/self.split/'cutting'/data['path']
        # assert path.is_file()
        waveform, sample_rate = torchaudio.load(path)
        sentence = data["sentence"]
        translation =  data["translation"]
        speaker_id = data["client_id"]
        _id = data["path"].replace(".wav", "")
        return waveform, sample_rate, sentence, translation, speaker_id, _id


def process(args):

    err_utt_id = []

    root = Path(args.data_root).absolute() 
    if not root.is_dir():
        raise NotADirectoryError(f"{root} does not exist")
    # Extract features
    feature_root = root / "fbank80"
    feature_root.mkdir(exist_ok=True)
    for split in SPLITS:
        print(f"Fetching split {split}...")
        dataset = BSTC(root, split)
        print("Extracting log mel filter bank features...")
        for waveform, sample_rate, _, _, _, utt_id in tqdm(dataset):
            try:
                extract_fbank_features(
                    waveform, sample_rate, feature_root / f"{utt_id}.npy"
                )
            except:
                err_utt_id.append(utt_id)
              
    # Pack features into ZIP
    zip_path = root / "fbank80.zip"
    print("ZIPing features...")
    create_zip(feature_root, zip_path)
    print("Fetching ZIP manifest...")
    zip_manifest = get_zip_manifest(zip_path)
    # Generate TSV manifest
    print("Generating manifest...")
    train_text = []
    task = f"asr_{args.src_lang}"
    if args.tgt_lang is not None:
        task = f"st_{args.src_lang}_{args.tgt_lang}"
    for split in SPLITS:
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        dataset = BSTC(root, split)
        for wav, sr, src_utt, tgt_utt, speaker_id, utt_id in tqdm(dataset):
            if utt_id in err_utt_id:
                continue
            manifest["audio"].append(zip_manifest[utt_id])
            manifest["id"].append(utt_id)
            
            duration_ms = int(wav.size(1) / sr * 1000)
            manifest["n_frames"].append(int(1 + (duration_ms - 25) / 10))
            manifest["tgt_text"].append(src_utt if args.tgt_lang is None else tgt_utt)
            manifest["speaker"].append(speaker_id)
        is_train_split = split.startswith("train")
        if is_train_split:
            train_text.extend(manifest["tgt_text"])
        df = pd.DataFrame.from_dict(manifest)
        df = filter_manifest_df(df, is_train_split=is_train_split)
        save_df_to_tsv(df, root / f"{split}_{task}.tsv")
    # Generate vocab
    vocab_type = args.src_voacb_type
    vocab_size = str(args.src_vocab_size)
    if args.tgt_lang is not None:
        vocab_type = args.trg_voacb_type
        vocab_size = str(args.trg_vocab_size)
    spm_filename_prefix = f"spm_{vocab_type}{vocab_size}_{task}"
    with NamedTemporaryFile(mode="w") as f:
        for t in train_text:
            f.write(t + "\n")
        gen_vocab(
            Path(f.name),
            root / spm_filename_prefix,
            vocab_type,
            vocab_size
        )
    # Generate config YAML
    gen_config_yaml(
        root,
        spm_filename_prefix + ".model",
        yaml_filename=f"config_{task}.yaml",
        specaugment_policy="lb",
    )
    # Clean up
    # shutil.rmtree(feature_root)

    print("%s:%s"%(split,err_utt_id)) # train:
    '''
    ['4385_136', '5_0', '5_3', '5_15', '5_18', '5_24', '5_54', '5_76', '5_87', '5_96', '5_122', '5_129', '5_132', '5_133', ...]
    '''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", "-d", type=str,
        help="data root with sub-folders for each language <root>/<src_lang>",
        default=r"/content/drive/Shareddrives/mahouli249@gmail.com/dataset/bstc/root"
    )
    parser.add_argument(
        "--src-vocab-type",
        default="char",
        type=str,
        choices=["bpe", "unigram", "char"],
    )
     parser.add_argument(
        "--trg-vocab-type",
        default="unigram",
        type=str,
        choices=["bpe", "unigram", "char"],
    )
    parser.add_argument("--src-vocab-size", default=3000, type=int)
    parser.add_argument("--trg-vocab-size", default=10000, type=int)
    parser.add_argument("--src-lang", "-s", type=str,default="ch")
    parser.add_argument("--tgt-lang", "-t", type=str)#,default="en"
    args = parser.parse_args()

    process(args)


if __name__ == "__main__":
    main()
