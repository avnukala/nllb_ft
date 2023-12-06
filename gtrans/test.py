import time
from googletrans import Translator
import pandas as pd
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate import bleu

df = pd.read_csv('../en_kr_data/val/ko2en_validation_csv/ko2en_medical_2_validation.csv', usecols=['한국어', '영어'])

korean_sentences = df['한국어'].tolist()[5000:]
english_sentences = df['영어'].tolist()[5000:]

english_translations = []
translator = Translator(service_urls=[
      'translate.google.com',
      'translate.google.co.kr',
    ])

try:
    translations = translator.translate(korean_sentences)
    english_translations = [x.text.split() for x in translations]
except:
    print("exception")

for i, k in tqdm(enumerate(korean_sentences)):
    if i%100 == 99: 
        print(corpus_bleu(english_sentences[:i], english_translations))
    try:
        translation = translator.translate(k)
        print(1)
        english_translations.append(translation.text.split())
        english_sentences.append([df.iloc[i]['영어'].split()])
    except:
        time.sleep(1)
        print('exception')

with open('../grans_out.txt', 'w') as of:
    for e in english_translations:
        of.write(e)

print(corpus_bleu(english_sentences, english_translations))