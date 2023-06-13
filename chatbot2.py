import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

import speech_recognition as sr
import pyttsx3

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json").read())

words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model("chatbotmodel.h5")

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

print("GO! Voice Assistant is running!")

while True:
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)

    try:
        message = recognizer.recognize_google(audio)
        print("User:", message)
        ints = predict_class(message)
        res = get_response(ints, intents)
        print("Assistant:", res)
        speak(res)
    except sr.UnknownValueError:
        print("Sorry, I didn't understand that. Could you please repeat?")
        speak("Sorry, I didn't understand that. Could you please repeat?")
    except sr.RequestError:
        print("Sorry, I'm having trouble processing your request. Please try again later.")
        speak("Sorry, I'm having trouble processing your request. Please try again later.")
