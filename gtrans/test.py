import time
from googletrans import Translator
import pandas as pd
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from nltk.translate import bleu

df = pd.read_csv('gtrans/data/val/ko2en_law_2_validation.csv', usecols=['한국어', '영어'])

korean_sentences = df['한국어'].tolist()[:500]

english_sentences = []
english_translations = []
translator = Translator()
for i, k in tqdm(enumerate(korean_sentences)):
    if i%100 == 99: 
        print(corpus_bleu(english_sentences[:i], english_translations))
    try:
        english_translations.append(translator.translate(k).text.split())
        english_sentences.append([df.iloc[i]['영어'].split()])
    except:
        time.sleep(20)
        print('exception')
        #english_translations.append(translator.translate(k).text.split())
        #english_sentences.append([df.iloc[i]['영어'].split()])
#translations = translator.translate(korean_sentences)

#english_sentences = [t.text for t in translations]
#english_sentences[:5]
print(corpus_bleu(english_sentences, english_translations))