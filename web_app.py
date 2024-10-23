from flask import Flask, render_template, request, jsonify
from voice_assistant import FridayAssistant
import random
import requests

app = Flask(__name__)
assistant = FridayAssistant()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_command', methods=['POST'])
def process_command():
    command = request.json['command']
    response = assistant.process_command(command)
    return jsonify({'response': response})

@app.route('/get_images')
def get_images():
    # In a real application, you would fetch these from a database or file system
    images = [
        'https://source.unsplash.com/random/1920x1080/?nature',
        'https://source.unsplash.com/random/1920x1080/?technology',
        'https://source.unsplash.com/random/1920x1080/?abstract',
        'https://source.unsplash.com/random/1920x1080/?space',
    ]
    return jsonify(images)

@app.route('/get_quote')
def get_quote():
    quotes = [
        "The only way to do great work is to love what you do. - Steve Jobs",
        "Innovation distinguishes between a leader and a follower. - Steve Jobs",
        "The future belongs to those who believe in the beauty of their dreams. - Eleanor Roosevelt",
        "Strive not to be a success, but rather to be of value. - Albert Einstein",
        "The only limit to our realization of tomorrow will be our doubts of today. - Franklin D. Roosevelt"
    ]
    return jsonify({'quote': random.choice(quotes)})

@app.route('/get_fun_fact')
def get_fun_fact():
    facts = [
        "The first computer programmer was a woman named Ada Lovelace.",
        "The world's first website is still online.",
        "The first computer mouse was made of wood.",
        "The first computer virus was created in 1983.",
        "The most common password is '123456'."
    ]
    return jsonify({'fact': random.choice(facts)})

@app.route('/get_weather')
def get_weather():
    # This is a mock weather API call. In a real application, you would use a proper weather API.
    weather_conditions = ['Sunny', 'Cloudy', 'Rainy', 'Snowy', 'Windy']
    temperature = random.randint(0, 40)
    condition = random.choice(weather_conditions)
    return jsonify({'temperature': temperature, 'condition': condition})

@app.route('/toggle_dark_mode', methods=['POST'])
def toggle_dark_mode():
    # This route doesn't actually change anything server-side,
    # but it provides an endpoint for the frontend to hit when toggling dark mode
    return jsonify({'success': True})

if __name__ == '__main__':
    app.run(debug=True)

if __name__ == '__main__':
    app.run(debug=True)