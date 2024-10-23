from datetime import datetime
import mysql.connector
import mysql

class DatabaseOperations:
    def __init__(self):
        self.connection = None
        self.cursor = None
        self.connect()

    def connect(self):
        try:
            self.connection = mysql.connector.connect(
                host="localhost",
                user="root",
                password="Yash@21",
                database="voice_assistant_db"
            )
            self.cursor = self.connection.cursor(dictionary=True)
            self.create_tables()
        except mysql.connector.Error as err:
            print(f"Error connecting to MySQL: {err}")
            print("Please make sure MySQL is running and the database 'voice_assistant_db' exists.")
            raise

    def reconnect(self):
        if not self.connection or not self.connection.is_connected():
            self.connect()

    def create_tables(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255),
                age VARCHAR(50),
                gender VARCHAR(50),
                hobbies TEXT,
                location VARCHAR(255)
            )
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT,
                timestamp DATETIME,
                user_input TEXT,
                assistant_response TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        self.connection.commit()

    def add_user(self, user_data):
        query = """
            INSERT INTO users (name, age, gender, hobbies, location)
            VALUES (%(name)s, %(age)s, %(gender)s, %(hobbies)s, %(location)s)
        """
        self.cursor.execute(query, user_data)
        self.connection.commit()

    def get_user(self, name):
        query = "SELECT * FROM users WHERE name = %s"
        self.cursor.execute(query, (name,))
        return self.cursor.fetchone()

    def get_all_users(self):
        query = "SELECT * FROM users"
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def save_conversation_history(self, user_name, conversation_history):
        user = self.get_user(user_name)
        if user:
            for entry in conversation_history:
                query = """
                    INSERT INTO conversation_history (user_id, timestamp, user_input, assistant_response)
                    VALUES (%s, %s, %s, %s)
                """
                self.cursor.execute(query, (user['id'], datetime.now(), entry['user'], entry['assistant']))
            self.connection.commit()

    def get_conversation_history(self, user_name):
        user = self.get_user(user_name)
        if user:
            query = """
                SELECT user_input, assistant_response
                FROM conversation_history
                WHERE user_id = %s
                ORDER BY timestamp DESC
                LIMIT 10
            """
            self.cursor.execute(query, (user['id'],))
            return [{'user': row['user_input'], 'assistant': row['assistant_response']} for row in self.cursor.fetchall()]
        return []

    def close(self):
        """Manually close the cursor and connection."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()



if __name__ == "__main__":
    db = DatabaseOperations()
    db.close()
    # Test database operations here