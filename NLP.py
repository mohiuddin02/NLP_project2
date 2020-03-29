# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 10:19:33 2020

@author: mohiu
"""

import nltk
nltk.download('stopwords')
nltk.download('punkt')
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))
import string
import re
import datetime
import time
from numpy import array, argmax, random, take
import pandas as pd
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Dense, LSTM, Embedding, Bidirectional, RepeatVector, TimeDistributed
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras import optimizers
import matplotlib.pyplot as plt
% matplotlib inline
pd.set_option('display.max_colwidth', 200)
from nltk.translate.bleu_score import sentence_bleu
# function to read raw text file
def read_text(filename):
    # open the file
    file = open(filename, mode='rt', encoding='utf-8')
    # read all text
    text = file.read()
    file.close()
    return text

# split a text into sentences
def to_lines(text):
    sents = text.strip().split('\n')
    sents = [i.split('\t') for i in sents]
    return sents

# function to build a tokenizer
def tokenization(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
    # integer encode sequences
    seq = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    seq = pad_sequences(seq, maxlen=length, padding='post')
    return seq

# build NMT model
def build_model(in_vocab, out_vocab, in_timesteps, out_timesteps, hidden_size):
    model = Sequential()
    model.add(Embedding(in_vocab, hidden_size, input_length=in_timesteps, mask_zero=True))
    model.add(LSTM(hidden_size))
    model.add(RepeatVector(out_timesteps))
    model.add(LSTM(hidden_size, return_sequences=True))
    model.add(Dense(out_vocab, activation='softmax'))
    return model

def get_word(n, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == n:
            return word
    return None

def calc_BLUE(pred_df):
    blue = 0
    for i in range(len(pred_df)):
        reference = [pred_df.iloc[i]['actual'].split()]
        candidate = pred_df.iloc[i]['predicted'].split()
        blue = blue + sentence_bleu(reference, candidate)
    return blue/len(pred_df)

def removeSentStopWords(sent):
  word_tokens = word_tokenize(sent.lower())
  filtered_sentence = [w for w in word_tokens if not w in stop_words]
  newSent = ''
  for s in range(len(filtered_sentence)):
    if s < len(filtered_sentence) - 1:
      if filtered_sentence[s+1].strip() in string.punctuation:
        newSent = newSent + filtered_sentence[s]
      else:
        newSent = newSent + filtered_sentence[s] + ' '
    else:
      newSent = newSent + filtered_sentence[s]
  return newSent

def RunAutoEncoder(usingNewsela=True, word_cutoff = 15, aggr_length = 12, trainTestSplitPerc = 0.3,
                   hidden_size = 64, remove_punc = True, batchSize=64,remove_stopwords = False, 
                   normFilePath='NewselaAlignedNorm-0.4.txt', simpFilePath='NewselaAlignedSimp-0.4.txt"'):
  if usingNewsela:
    data_normal = read_text(normFilePath)
    norm_data = to_lines(data_normal)
    norm_data = array(norm_data)
    norm_data = [s[0] for s in norm_data]

    data_simple = read_text(simpFilePath)
    simp_data = to_lines(data_simple)
    simp_data = array(simp_data)
    simp_data = [s[0] for s in simp_data]
  else:
    data_normal = read_text(normFilePath)
    norm_data = to_lines(data_normal)
    norm_data = array(norm_data)
    norm_data = [s[2] for s in norm_data]

    data_simple = read_text(simpFilePath)
    simp_data = to_lines(data_simple)
    simp_data = array(simp_data)
    simp_data = [s[2] for s in simp_data]
  
  
  #Remove stop words
  if remove_stopwords:
    print("Removving stop words...")
    for n in range(len(norm_data)):
      norm_data[n] = removeSentStopWords(norm_data[n])
    for s in range(len(simp_data)):
      simp_data[n] = removeSentStopWords(simp_data[n])
  allData= []
  allData_aggr=[]
  for i in range(len(norm_data)):
      if (len(norm_data[i].split()) < word_cutoff) and (len(simp_data[i].split()) < len(norm_data[i].split())):
          allData.append([norm_data[i], simp_data[i]])
  
  allData = array(allData)  


  # Remove punctuation
  if remove_punc:
    print("Removing punctuations...")
    allData[:,0] = [s.translate(str.maketrans('', '', string.punctuation)) for s in allData[:,0]]
    allData[:,1] = [s.translate(str.maketrans('', '', string.punctuation)) for s in allData[:,1]]

  # convert to lowercase
  for i in range(len(allData)):
      allData[i,0] = allData[i,0].lower().strip()
      
      allData[i,1] = allData[i,1].lower().strip()
      allData_aggr.append(allData[i,0])
      allData_aggr.append(allData[i,1])

  # empty lists
  norm_l = []
  simp_l = []

  # populate the lists with sentence lengths
  for i in allData[:,0]:
      norm_l.append(len(i.split()))

  for i in allData[:,1]:
      simp_l.append(len(i.split()))

  # prepare aggregate tokenizer
  aggr_tokenizer = tokenization(allData_aggr)
  aggr_vocab_size = len(aggr_tokenizer.word_index) + 1
  # prepare normal tokenizer
  norm_tokenizer = tokenization(allData[:, 0])
  norm_vocab_size = len(norm_tokenizer.word_index) + 1
  # prepare simple tokenizer
  simp_tokenizer = tokenization(allData[:, 1])
  simp_vocab_size = len(simp_tokenizer.word_index) + 1
  
  train, test = train_test_split(allData, test_size=trainTestSplitPerc, random_state = 12)
  
  # prepare training data
  trainX = encode_sequences(aggr_tokenizer, aggr_length, train[:, 0])
  trainY = encode_sequences(aggr_tokenizer, aggr_length, train[:, 1])

  # prepare validation data
  testX = encode_sequences(aggr_tokenizer, aggr_length, test[:, 0])
  testY = encode_sequences(aggr_tokenizer, aggr_length, test[:, 1])

  model = build_model(aggr_vocab_size, aggr_vocab_size, aggr_length, aggr_length, hidden_size)
  rms = optimizers.RMSprop(lr=0.001)
  model.compile(optimizer=rms, loss='sparse_categorical_crossentropy')

  filename = 'NLPFinalModel'
  checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
  start_time = time.time()
  history = model.fit(trainX, trainY.reshape(trainY.shape[0], trainY.shape[1], 1), 
            epochs=50, batch_size=batchSize, 
            validation_split = 0.2,
            callbacks=[checkpoint], verbose=1)
  print("--- %s seconds ---" % (time.time() - start_time))

  model.summary()
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.legend(['train','validation'])
  plt.xlabel('# epochs')
  plt.ylabel('Loss')
  plt.title('Model training')
  plt.show()

  model = load_model('NLPFinalModel')
  preds = model.predict_classes(testX.reshape((testX.shape[0],testX.shape[1])))

  # convert predictions into text (English)
  preds_text = []
  for i in preds:
      temp = []
      for j in range(len(i)):
          t = get_word(i[j], aggr_tokenizer)
          if j > 0:
              if (t == get_word(i[j-1], aggr_tokenizer)) or (t == None):
                  temp.append('')
              else:
                  temp.append(t)
              
          else:
              if(t == None):
                  temp.append('')
              else:
                  temp.append(t)            
          
      preds_text.append(' '.join(temp))
    
  print(calc_BLUE(pd.DataFrame({'actual' : test[:,0], 'predicted' : preds_text})))
  
#user variables
#simple wikipedia data
#normFilePath = "alignedSentencesNorm-0.3.txt"
#simpFilePath = "alignedSentencesSimp-0.3.txt"

#newsela corpus
normFilePath = "alignedSentencesNorm-0.3.txt"
simpFilePath = "alignedSentencesSimp-0.3.txt"

usingNewsela = True
remove_punc = True
remove_stopwords = False
word_cutoff = 15
aggr_length = 15
hidden_size = 256
batchSize = 64
trainTestSplitPerc= 0.3
RunAutoEncoder(usingNewsela, word_cutoff, aggr_length, trainTestSplitPerc, hidden_size, remove_punc, batchSize, remove_stopwords, normFilePath, simpFilePath)