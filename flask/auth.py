from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_mysqldb import MySQL

app = Flask(__name__)  # Fix: Use __name__ instead of __auth__
CORS(app)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000/login"}})

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'user'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

mysql = MySQL(app)

@app.route('/login', methods=['POST','OPTIONS'])
def login():
    data = request.get_json()
    name = data.get('username')
    password = data.get('password')

    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM users WHERE name = %s AND password = %s", (name, password))  # Fix: Use %s for password
    user = cur.fetchone()
    cur.close()

    if user:
        return jsonify({'message': 'Login successful'})
    else:
        return jsonify({'message': 'Invalid credentials'}), 401

if __name__ == '__auth__':
    app.run(debug=True)
