import nltk
import json
import random
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import speech_recognition as sr
import os

from tkinter import *
from tkinter import scrolledtext

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# load the trained model, lemmatizer, intents, words, and classes
model = load_model('chatbot_model_v2.h5')
lemmatizer = WordNetLemmatizer()
intents = json.loads(open("E:/woxsen/Deep Learning/course project_chatbot/project/intents.json").read())
words = pickle.load(open('words_v2.pkl','rb'))
classes = pickle.load(open('classes_v2.pkl','rb'))

# clean up the user's sentence by tokenizing and lemmatizing
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# create a bag of words array for the user's sentence
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

# predict the intent of the user's sentence using the trained model
def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# get a response from the intents file based on the predicted intent
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# generate a response from the chatbot given a user input
def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

# create a recognizer object
r = sr.Recognizer()

# create a GUI window
root = Tk()
root.title("Chatbot")

# create a text box for the chat history
history = scrolledtext.ScrolledText(root, width=60, height=20)
history.grid(column=0, row=0, padx=10, pady=10)

# create a text box for the user input
user_input = Entry(root, width=60)
user_input.grid(column=0, row=1, padx=10, pady=10)

# function to get the user's message and display the chat history
def enter_pressed(event):
    input_text = user_input.get()
    history.insert(INSERT, "You: " + input_text + "\n")
    user_input.delete(0, END)
    response = chatbot_response(input_text)
    history.insert(INSERT, "Bot: " + response + "\n")
