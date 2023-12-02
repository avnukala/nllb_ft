import argparse
import logging
import random
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
import random
from copy import deepcopy
from tqdm.auto import trange

def get_batch_pairs(batch_size, data, titles=[('영어', 'en_Latn'), ('한국어', 'kor_Hang')]):
    (l1, long1), (l2, long2) = random.sample(titles, 2)
    x, y = [], []
    for _ in range(batch_size):
        item = data.iloc[random.randint(0, len(data)-1)]
        x.append(item[l1])
        y.append(item[l2])
    return x, y, long1, long2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch_size', default=256, type=int,
                    help='batch training size')
    ap.add_argument('--max_length', default=60, type=int,
                    help='max sentence length')
    ap.add_argument('--n_iters', default=10000, type=int,
                    help='total number of examples to train on')
    ap.add_argument('--print_every', default=500, type=int,
                    help='print loss info every this many training examples')
    ap.add_argument('--checkpoint_every', default=1000, type=int,
                    help='write out checkpoint every this many training examples')
    ap.add_argument('--initial_learning_rate', default=0.001, type=float,
                    help='initial learning rate')
    ap.add_argument('--src_lang', default='fr',
                    help='Source (input) language code, e.g. "fr"')
    ap.add_argument('--tgt_lang', default='en',
                    help='Source (input) language code, e.g. "en"')
    ap.add_argument('--train_path', default='en_kr_data/train/ko2en_training_csv/ko2en_finance_1_training.csv',
                    help='training file')
    ap.add_argument('--val_path', default='en_kr_data/val/ko2en_validation_csv/ko2en_finance_2_validation.csv',
                    help='val file')
    ap.add_argument('--out_file', default='out.txt',
                    help='output file')
    ap.add_argument('--save_path', default='/models',
                    help='folder to save models')
    ap.add_argument('--load_checkpoint', nargs=1,
                    help='checkpoint file to start from')

    args = ap.parse_args()

    train_df = pd.read_csv(args.train_path, sep=',')
    val_test_df = pd.read_csv(args.val_path, sep=',')
    val_df, test_df = val_test_df[:len(val_test_df)//2], val_test_df[len(val_test_df)//2:]

    if args.load_checkpoint is not None:
        pass
    else:
        name = "facebook/nllb-200-distilled-600M"
        tokenizer = AutoTokenizer.from_pretrained(name)
        nllb = AutoModelForSeq2SeqLM.from_pretrained(name)
    
    tokenizer.src_lang = "kor_Hang"
    inputs = tokenizer(text="다만 하반기에도 이같은 기조가 이어질 지에 대해서는 의견이 분분하다.", return_tensors="pt")
    translated_tokens = nllb.generate(
        **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"]
    )

    batch_size = 16
    test_df_small = test_df[:1000]
    #batches = [test_df.iloc[i:i + batch_size] for i in range(0, len(test_df), batch_size)]
    batches = [test_df_small.iloc[i:i + batch_size] for i in range(0, len(test_df_small), batch_size)]

    # test original
    english_translations = []
    for df_batch in tqdm(batches):
        inputs = tokenizer(text=df_batch['한국어'].tolist(), return_tensors="pt", padding=True, truncation=True)
        translated_tokens = nllb.generate(
            **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"]
        )
        translated_sentences = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        #print(translated_sentences)
        english_translations += translated_sentences
    english_translations_split = [t.split() for t in english_translations]
    english_sentences = test_df['영어'].tolist()[:1000]
    english_sentences_split = [[s.split()] for s in english_sentences]
    logging.info('Original model BLEU score: %.3f', corpus_bleu(english_sentences_split, english_translations_split))

    from transformers.optimization import Adafactor
    from transformers import get_constant_schedule_with_warmup
    #nllb.cuda();
    optimizer = Adafactor(
        [p for p in nllb.parameters() if p.requires_grad],
        scale_parameter=False,
        relative_step=False,
        lr=1e-4,
        clip_threshold=1.0,
        weight_decay=1e-3,
    )
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=100)

    data_train = train_df.copy()
    data_train = data_train.filter(['한국어', '영어'], axis=1)

    losses = []
    nllb_new = deepcopy(nllb)
    nllb_new.train()
    x, y, loss = None, None, None

    tq = trange(len(losses), args.n_iters)
    for i in tq:
        x, y, lang1, lang2 = get_batch_pairs(batch_size)
        try:
            tokenizer.src_lang = lang1
            x = tokenizer(x, return_tensors='pt', padding=True, truncation=True, max_length=args.max_length).to(nllb_new.device)
            tokenizer.src_lang = lang2
            y = tokenizer(y, return_tensors='pt', padding=True, truncation=True, max_length=args.max_length).to(nllb_new.device)
            y.input_ids[y.input_ids == tokenizer.pad_token_id] = -100

            loss = nllb_new(**x, labels=y.input_ids).loss
            loss.backward()
            losses.append(loss.item())

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        except: 
            optimizer.zero_grad(set_to_none=True)
            x, y, loss = None, None, None
            print('exception')
            continue

        if i % 1000 == 0:
            print(i, np.mean(losses[-1000:]))

        if i % 1000 == 0 and i > 0:
            nllb_new.save_pretrained(args.save_path)
            tokenizer.save_pretrained(args.save_path)



if __name__ == '__main__':
    main()