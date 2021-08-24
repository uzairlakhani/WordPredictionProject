from django.shortcuts import render


from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow.keras.utils as ku 
import numpy as np 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
import nltk
import pickle

# Create your views here.

t = Tokenizer()
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

model = load_model('NextWordPredictorModel1(letter)2.h5',compile = False)

def predictword(text, num_words):
    num_spaces = 0
    text = text.lower()
    values = []
    for i in range(len(text)):
        new_item1 = tokenizer[text[i]]
        values.append(new_item1)
    
    while(num_spaces <= num_words):
        token_list = t.texts_to_sequences([text])[0]
        token_list = pad_sequences([values], 100, padding='pre')
        predicted = model.predict(token_list)
        index = np.argmax(predicted)
        if(index == 0):
            num_spaces +=1
        values.append(index)
    
    key_list = list(tokenizer.keys())
    val_list = list(tokenizer.values())
    
    predicted_list = []
    
    for i in range(len(values)):
        predicted_list.append(key_list[values[i]])
    
    text="".join(predicted_list)
    text = text.split()
    text = text[-num_words:]
    text = " ".join(text)

    return text


def prediction(request):
  predict1 = request.POST.get('predict')
  initial_value = request.POST.get('predict')
  predict1 = predictword(predict1, 1)
  if initial_value == "":
    predict1 = ""
  

  context = {"initial": initial_value, "predict": predict1}
  return render(request, "WordPredictionApp/prediction.html", context)