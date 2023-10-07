from flask import Flask, request, jsonify
from flask_cors import CORS
import pymongo
import os

app = Flask(__name__)
CORS(app, supports_credentials=True)
# Define the directory where uploaded images will be stored
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# MongoDB setup

client = pymongo.MongoClient('mongodb+srv://bsse1130:11811109@cluster0.lb7vjxi.mongodb.net/')

db = client['Sample']

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # Check if the 'file' key is in the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        # Check if the file is empty
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        # Check if the file is allowed (you can add more file types if needed)
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
        if not allowed_file(file.filename, allowed_extensions):
            return jsonify({'error': 'Invalid file type'})

        # Save the uploaded file to the upload folder
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

        # Store file information in MongoDB
        file_info = {
            'filename': file.filename,
            'file_path': os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        }
        print(file.filename)
        db['Id'].insert_one(file_info)

        return jsonify({'message': 'File uploaded successfully'})

    except Exception as e:
        return jsonify({'error': str(e)})

def allowed_file(filename, allowed_extensions):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

if __name__ == '__main__':
    app.run(debug=True)
