import nltk
import json
import random
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import speech_recognition as sr
import os

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

start = True
while start:
    # ask user to speak or type a message
    mode = input('Enter "s" to speak, or "t" to type: ')
    
    if mode == 's':
        # use microphone as audio source
        with sr.Microphone() as source:
            print("Listening...")
            audio = r.listen(source)
        try:
            query = r.recognize_google(audio)
            print("You said:", query)
            ints = predict_class(query, model)
            res = getResponse(ints, intents)
            print(res + '\n')
        except sr.UnknownValueError:
            print("Sorry, I didn't understand that. Please try again.\n")
     
    elif mode == 't':
        query = input('Enter your message: ')
        
    # get a response from the chatbot based on the user's input
    response = chatbot_response(query)
    print
