import argparse
import logging
import random
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, NllbTokenizerFast, get_constant_schedule_with_warmup
from transformers.optimization import Adafactor
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
import gc
import matplotlib.pyplot as plt

def get_batch_pairs(batch_size, data, titles=[('영어', 'en_Latn'), ('한국어', 'kor_Hang')]):
    (l1, long1), (l2, long2) = random.sample(titles, 2) # randomly choose translation direction
    x, y = [], []
    for _ in range(batch_size):
        item = data.iloc[random.randint(0, len(data)-1)]
        x.append(item[l1])
        y.append(item[l2])
    return x, y, long1, long2

def clear_mem():
  gc.collect()
  torch.cuda.empty_cache()
  
def plot_loss_and_bleu(losses, bleu_scores, iters, save_path):
  plt.plot(losses)
  plt.xlabel("Iterations")
  plt.ylabel("Loss")
  plt.title(str(iters)+" Iterations")
  plt.show()
  plt.savefig(save_path+'/train_loss.png')
  
  plt.plot(bleu_scores)
  plt.xlabel("Iterations")
  plt.ylabel("Bleu Score on Test")
  plt.title(str(iters)+" Iterations")
  plt.show()
  plt.savefig(save_path+'/bleu_score.png')
  
def sort_dataset_by_size(data):
    return data.sort_values(by=['한국어_어절수'])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch_size', default=256, type=int,
                    help='batch training size')
    ap.add_argument('--max_length', default=60, type=int,
                    help='max sentence length')
    ap.add_argument('--n_iters', default=5000, type=int,
                    help='total number of examples to train on')
    ap.add_argument('--print_every', default=500, type=int,
                    help='print loss info every this many training examples')
    ap.add_argument('--checkpoint_every', default=1000, type=int,
                    help='write out checkpoint every this many training examples')
    ap.add_argument('--plot_every', default=1000, type=int,
                    help='plot this many training examples')
    ap.add_argument('--initial_learning_rate', default=0.001, type=float,
                    help='initial learning rate')
    ap.add_argument('--train_path', default='en_kr_data/train/ko2en_training_csv/ko2en_medical_1_training.csv',
                    help='training file')
    ap.add_argument('--val_path', default='en_kr_data/val/ko2en_validation_csv/ko2en_medical_2_validation.csv',
                    help='val file')
    ap.add_argument('--out_file', default='out.txt',
                    help='output file')
    ap.add_argument('--model_save_path', default='/models',
                    help='folder to save models')
    ap.add_argument('--plot_save_path', default='/plots',
                    help='folder to save plots')
    ap.add_argument('--load_checkpoint', default=None,
                    help='checkpoint file to start from')
    ap.add_argument('--curriculum', '-c', action='store_true', 
                    help='curriculum learning (process data by size)')
    ap.add_argument('--lr', default=1e-4,
                    help='learning rate')
    

    args = ap.parse_args()

    train_df = pd.read_csv(args.train_path, sep=',')
    train_df = sort_dataset_by_size(train_df)

    val_test_df = pd.read_csv(args.val_path, sep=',')
    val_df, test_df = val_test_df[:len(val_test_df)//2], val_test_df[len(val_test_df)//2:]

    if args.load_checkpoint is not None:
        tokenizer = NllbTokenizerFast.from_pretrained(args.load_checkpoint)
        nllb = AutoModelForSeq2SeqLM.from_pretrained(args.load_checkpoint)
    else:
        name = "facebook/nllb-200-distilled-600M"
        tokenizer = NllbTokenizerFast.from_pretrained(name)
        nllb = AutoModelForSeq2SeqLM.from_pretrained(name)

    def test_translations(model, data, batch_size=25, num_beams=4, small=False):
        if small:
            data = data[:len(data)//10]
        korean_src = data['한국어'].tolist()
        batches = [korean_src[i:i + batch_size] for i in range(0, len(korean_src), batch_size)]

        english_translations = []
        for df_batch in tqdm(batches):
            tokenizer.src_lang = 'kor_Hang'
            tokenizer.tgt_lang = 'eng_Latn'
            inputs = tokenizer(text=df_batch['한국어'].tolist(), return_tensors="pt", padding=True, truncation=True)
            model.eval()
            translated_tokens = model.generate(
                **inputs.to(model.device), 
                forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"],
                num_beams=4
            )
            translated_sentences = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
            english_translations += translated_sentences
        english_translations_split = [t.split() for t in english_translations]
        english_gt_split = [[s.split()] for s in data['영어'].tolist()]
        bleu_score = corpus_bleu(english_gt_split, english_translations_split)
        print('Model BLEU score: %.3f', bleu_score)

        with open(args.out_file, 'w') as of:
            for e in english_translations:
                of.write(e)

        return bleu_score

    optimizer = Adafactor(
        [p for p in nllb.parameters() if p.requires_grad],
        scale_parameter=False,
        relative_step=False,
        lr=args.lr,
        clip_threshold=1.0,
        weight_decay=1e-3,
    )
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=100)

    data_train = train_df.copy()
    
    if args.curriculum:
      data_train = sort_dataset_by_size(data_train)
      
    data_train = data_train.filter(['한국어', '영어'], axis=1)

    batch_size = args.batch_size
    losses = []
    bleu_scores = []
    nllb_new = deepcopy(nllb)
    nllb_new.train()
    x, y, loss = None, None, None

    tq = trange(len(losses), args.n_iters)
    for i in tq:
        x, y, lang1, lang2 = get_batch_pairs(batch_size, data_train)
        try:
            tokenizer.src_lang = lang1
            x = tokenizer(x, return_tensors='pt', padding=True, truncation=True, max_length=args.max_length).to(nllb_new.device)
            tokenizer.src_lang = lang2
            y = tokenizer(y, return_tensors='pt', padding=True, truncation=True, max_length=args.max_length).to(nllb_new.device)
            y.input_ids[y.input_ids == tokenizer.pad_token_id] = -100

            loss = nllb_new(**x, labels=y.input_ids).loss
            loss.backward()
            losses.append(loss.item())

            if i % args.print_every == 0:
                print(loss.item())

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        except: 
            optimizer.zero_grad(set_to_none=True)
            x, y, loss = None, None, None
            clear_mem()
            continue

        if i % args.plot_every == 0:
            print(f'iteration: {i} mean loss: {np.mean(losses[-args.plot_every:])}')
            score = test_translations(nllb_new, val_df)
            print(f'current bleu score is: {score}')
            bleu_scores.append(score)
            plot_loss_and_bleu(losses, bleu_scores, i)

        if i % args.checkpoint_every == 0 and i > 0:
            nllb_new.save_pretrained(args.save_path)
            tokenizer.save_pretrained(args.save_path)

        
    final_bleu = test_translations(nllb_new, test_df)
    print(f'final bleu score is: {final_bleu}')
    

if __name__ == '__main__':
    main()