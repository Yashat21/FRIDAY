import logging
import bs4
import cv2
import numpy as np
import requests
import speech_recognition as sr
import pyttsx3
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import spacy
from user_management import UserManager
from database_operations import DatabaseOperations
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import datetime 
import random
import requests
import geocoder
from geopy.geocoders import Nominatim
import tensorflow as tf 
import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class FridayAssistant:
    def get_latest_news(self):
        try:
            url = "https://timesofindia.indiatimes.com/"
            response = requests.get(url)
            soup = bs4.BeautifulSoup(response.text, 'xml')
            news_items = soup.findAll('item')[:5]  
            
            news_summary = "Here are the latest news headlines:\n"
            for item in news_items:
                news_summary += f"- {item.title.text}\n"
            
            return news_summary
        except Exception as e:
            logging.error(f"Error fetching news: {str(e)}")
            return "I'm sorry, but I couldn't fetch the latest news at the moment. Please try again later."    
    def open_camera(self):
        try:
            # Initialize the camera
            self.camera = cv2.VideoCapture(0)
            
            if not self.camera.isOpened():
                return "Failed to open the camera. Please check if it's connected properly."

            # Read a frame from the camera
            ret, frame = self.camera.read()
            if not ret:
                return "Failed to capture image from the camera."

            # Display the frame
            cv2.imshow('Camera', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Release the camera
            self.camera.release()

            return "Camera opened successfully and image displayed."

        except Exception as e:
            return f"An error occurred while trying to open the camera: {str(e)}"

    def classify_image(self):
        ret, frame = self.camera.read()
        if not ret:
            return "Failed to capture image."
        
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        self.image_classifier.setInput(blob)
        detections = self.image_classifier.forward()
        
        results = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")
                label = f"{self.labels[idx]}: {confidence * 100:.2f}%"
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                results.append(label)
        
        cv2.imshow('Classified Image', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return f"I see: {', '.join(results)}" if results else "I couldn't identify any objects."

    def get_current_location(self):
        try:
            # Initialize the Nominatim geocoder
            geolocator = Nominatim(user_agent="friday_assistant")

            # Get the IP address
            ip_response = requests.get('https://api.ipify.org')
            if ip_response.status_code == 200:
                ip_address = ip_response.text
            else:
                return "I couldn't determine your IP address."

            # Get location using the IP address
            location = geolocator.geocode(ip_address)
            
            if location:
                address = location.address
                latitude = location.latitude
                longitude = location.longitude
                
                # Prepare the response
                response = f"Your current location is: {address}. The latitude is {latitude} and the longitude is {longitude}."
                
                # Speak the response
                self.speak(response)
                
                return response
            else:
                return "I couldn't determine your current location."
        except Exception as e:
            return f"An error occurred while fetching your location: {str(e)}"


    def get_weather(self):
        try:
            location = self.geolocator.geocode("10.10.193.182", timeout=10)
            if location:
                params = {
                    'key': self.weather_api_key,
                    'q': f"{location.latitude},{location.longitude}"
                }
                response = requests.get(self.weather_base_url, params=params)
                data = response.json()
                if 'current' in data:
                    temp = data['current']['temp_c']
                    condition = data['current']['condition']['text']
                    humidity = data['current']['humidity']
                    return f"Current weather in {location.address}: {condition}, Temperature: {temp}°C, Humidity: {humidity}%"
                else:
                    return "I couldn't fetch the weather information."
            else:
                return "I couldn't determine your location to fetch weather information."
        except Exception as e:
            return f"An error occurred while fetching weather information: {str(e)}"
        
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        self.nlp = spacy.load("en_core_web_sm")
        self.user_manager = UserManager()
        self.db_ops = DatabaseOperations()

        nltk.download('vader_lexicon', quiet=True)
        self.sia = SentimentIntensityAnalyzer()

        # Initialize the general sentiment analysis pipeline
        self.hf_sentiment = pipeline("sentiment-analysis")

        # Initialize the financial sentiment analysis pipeline
        model_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
        self.financial_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.financial_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.financial_sentiment = pipeline("sentiment-analysis", model=self.financial_model, tokenizer=self.financial_tokenizer)

        # Set voice to female
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[1].id)
        
        # Adjust voice properties for more natural sound
        self.engine.setProperty('rate', 150)
        self.engine.setProperty('volume', 1.0)

        self.name = "FRIDAY"
        self.conversation_history = []
        self.current_user = None

        # Initialize geolocator for weather and location services
        self.geolocator = Nominatim(user_agent="friday_assistant")

        # Initialize camera and image classifier (if needed)
        self.camera = None
        self.image_classifier = None
        self.labels = None  # You might want to load labels for image classification

        # Weather API setup (you need to set up your API key)
        self.weather_api_key = "YOUR_WEATHER_API_KEY"
        self.weather_base_url = "http://api.weatherapi.com/v1/current.json"

    def listen_for_wake_word(self):
        with sr.Microphone() as source:
            print("Listening for 'Hey Friday'...")
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source)
            try:
                text = self.recognizer.recognize_google(audio).lower()
                return "hey friday" in text
            except:
                return False

    def listen(self):
        with sr.Microphone() as source:
            print("Listening...")
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source)
            try:
                return self.recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                return "Sorry, I didn't catch that."

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def analyze_sentiment(self, text):

        vader_sentiment = self.sia.polarity_scores(text)
    
   
        hf_result = self.hf_sentiment(text)[0]
    
   
        financial_result = self.financial_sentiment(text)[0]
    
   
        vader_compound = vader_sentiment['compound']
        hf_score = hf_result['score'] if hf_result['label'] == 'POSITIVE' else -hf_result['score']
        financial_score = financial_result['score'] if financial_result['label'] == 'POSITIVE' else -financial_result['score']
        
        combined_sentiment = (vader_compound + hf_score + financial_score) / 3
        
        score = abs(combined_sentiment)
        
        return combined_sentiment, score


    def respond_based_on_sentiment(self, text):
        sentiment, score = self.analyze_sentiment(text)
        if sentiment == 'POSITIVE' and score > 0.6:
            return random.choice([
                "I'm glad you're feeling good!",
                "That's wonderful to hear!",
                "Your positive energy is contagious!",
            ])
        elif sentiment == 'NEGATIVE' and score > 0.6:
            return random.choice([
                "I'm sorry you're feeling upset. Is there anything I can do to help?",
                "It sounds like you're having a tough time. Would you like to talk about it?",
                "I'm here for you. What can I do to make your day better?",
            ])
        else:
            return random.choice([
                "I understand. How can I assist you further?",
                "Okay. What would you like to do next?",
            ])

    def process_command(self, command):
        doc = self.nlp(command)
    
    
        entities = [(ent.text, ent.label_) for ent in doc.ents]
    
  
        if "thank you friday" in command.lower():
            return "exit"
        elif any(token.text.lower() in ["weather", "temperature", "forecast"] for token in doc):
            return self.get_weather()
        elif any(token.text.lower() in ["time", "date", "day"] for token in doc):
            return f"The current time is {datetime.datetime.now().strftime('%I:%M %p')} and today is {datetime.datetime.now().strftime('%A, %B %d, %Y')}."
        elif any(token.text.lower() in ["joke", "funny"] for token in doc):
            return random.choice([
            "Why don't scientists trust atoms? Because they make up everything!",
            "Why did the scarecrow win an award? He was outstanding in his field!",
            "Why don't eggs tell jokes? They'd crack each other up!",
        ])
        elif "news" in command.lower():
            return self.get_latest_news()
        elif "camera" in command.lower() or "take picture" in command.lower():
            return self.open_camera()
        elif "classify image" in command.lower():
            return self.classify_image()
        elif "location" in command.lower():
            return self.get_current_location()
        else:
       
            for ent in entities:
                if ent[1] == "PERSON":
                    return f"I'm not sure how to help with that, {ent[0]}. Could you please clarify?"
        
        
        triples = []
        for token in doc:
            if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                subject = token.text
                verb = token.head.text
                obj = [child.text for child in token.head.children if child.dep_ == "dobj"]
                if obj:
        
                    if triples:
                        return f"I understand that {triples[0][0]} {triples[0][1]} {triples[0][2]}. How can I help with that?"
        
        response = self.respond_based_on_sentiment(command)
        self.conversation_history.append({"user": command, "assistant": response})
        return response

    def run(self):
        self.speak(f"Hello, I'm {self.name}. May I know your good name please  ?")
        name = self.listen()
        user = self.user_manager.get_user(name)
        
        if user:
            self.speak(f"Welcome back, {name}! How can I assist you today?")
            self.current_user = name
            self.conversation_history = self.db_ops.get_conversation_history(name)
            if self.conversation_history:
                last_conversation = self.conversation_history[-1]
                self.speak(f"Last time we talked about {last_conversation['user_id']}. Would you like to continue from there?")
        else:
            self.speak("I don't have your information in my database. Would you like to register?")
            response = self.listen()
            if "yes" in response.lower():
                self.current_user = self.user_manager.register_new_user()
            else:
                self.speak("Alright, I'll continue without personalizing my responses.")

        while True:
            if self.listen_for_wake_word():
                self.speak(f"Hello! How can I help you?")
                command = self.listen()
                response = self.process_command(command)
                if response == "exit":
                    self.speak("You're welcome! Have a great day!")
                    if self.current_user:
                        self.db_ops.save_conversation_history(self.current_user, self.conversation_history)
                    break
                self.speak(response)

if __name__ == "__main__":
    assistant = FridayAssistant()
    assistant.run()