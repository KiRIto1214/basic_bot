import random
import json
import numpy as np
import pickle

import nltk

from nltk.stem import WordNetLemmatizer


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense ,Activation ,Dropout
from tensorflow.keras.optimizers import SGD

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intent.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl','rb'))

model = load_model('chatbotmodel.h5')


def clean(sentence) :

    sentence_words = nltk.word_tokenize(sentence)

    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]

    return sentence_words

def bag_of_words(sentence) :


    sentence_words = clean(sentence)

    bag = [0]*len(words)

    for word in sentence_words :
        for i , j in enumerate(words) :

            if j == word :
                bag[i] = 1


    return np.array(bag)

def predict_class(sentence) :

    bow = bag_of_words(sentence)

    res = model.predict(np.array([bow]))[0]

    result = [[i,r] for i,r in enumerate(res) if r > 0.25]

    result.sort(key=lambda x : x[1],reverse =True)

    return_list = []

    for r in result :
        return_list.append({'intent' : classes[r[0]], 'probability' : str(r[1])})

    return return_list


def get_res(intent_list , intent_json) :
    tag = intent_list[0]['intent']

    list_of_intent = intent_json['intent']

    for i in list_of_intent :
        if i['tag'] == tag :
            result = random.choice(i['responses'])
            break

    return result

print("Bot is Online")
while True :
    message = input("")
    ints = predict_class(message)
    res = get_res(ints, intents)

    print(res)



    
    
    
    
