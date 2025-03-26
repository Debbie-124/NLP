import nltk
import numpy as np
nltk.download('punkt')  #這是已經預訓練好的斷詞處理模型
nltk.download('punkt_tab')
from nltk.stem.porter import PorterStemmer  #詞幹提取工具
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# #測試斷詞處理
# sentence_test = "My name is steven"
# print(sentence_test)
# sentence_tokenized = tokenize(sentence_test)
# print(sentence_tokenized)

def stem(word):
    return stemmer.stem(word.lower())

# 測試詞幹提取+轉乘小寫
# words = ["organize", "ORganizes", "orgaNizing"]
# print(words)
# stemmed_words = [stem(w) for w in words]
# print(stemmed_words)


def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype = np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence: 
            bag[idx] = 1.0
    return bag


#測試詞袋模型(BOW)
# sentence = ["hello","how","are","you"]
# words = ["hi","hello","I","you","bye","thank","cool"]
# bag = bag_of_words(sentence, words)
# print(sentence)
# print(words)
# print(bag)