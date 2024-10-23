import speech_recognition as sr
import pyttsx3
from database_operations import DatabaseOperations

class UserManager:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        self.db_ops = DatabaseOperations()

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def listen(self):
        with sr.Microphone() as source:
            print("Listening...")
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source)
            try:
                return self.recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                return "Sorry, I didn't catch that."

    def get_user(self, name):
        return self.db_ops.get_user(name)

    def register_new_user(self):
        self.speak("Let's register you in the system. What's your name?")
        name = self.listen()
        self.speak(f"Hello {name}. What's your age?")
        age = self.listen()
        self.speak("What's your gender?")
        gender = self.listen()
        self.speak("What are your hobbies?")
        hobbies = self.listen()
        self.speak("Where do you live?")
        location = self.listen()

        user_data = {
            "name": name,
            "age": age,
            "gender": gender,
            "hobbies": hobbies,
            "location": location
        }

        self.db_ops.add_user(user_data)
        self.speak("Thank you for registering. Your information has been saved.")
        self.speak("Do you want to access the database")
        password = self.listen()
        if password.lower()=="yes":
            self.speak("Please enter the password to view all user information.")
        password = self.listen()
        if password.lower() == "yashwanth":
            users = self.db_ops.get_all_users()
            for user in users:
                print(user)
                self.speak(f"User: {user['name']}, Age: {user['age']}, Gender: {user['gender']}, Hobbies: {user['hobbies']}, Location: {user['location']}")
        else:
            self.speak("Incorrect password. Access denied.")
            


    
        

if __name__ == "__main__":
    user_manager = UserManager()
    