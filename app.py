from flask import Flask, request, jsonify, render_template_string
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import json
import base64
from threading import Thread

app = Flask(__name__)

# Load and compile the trained model
model = tf.keras.models.load_model("model/model.h5")
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

img_height = 80
img_width = 60

with open('class_names.json', 'r') as f:
    class_names = json.load(f)


def preprocess_image(image):
    image = image.resize((img_width, img_height))
    image = np.array(image)
    image = image / 255.0
    return image

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        # Receive the request data
        data = request.json
        serialized_images_data = data['images']

        images = []
        filenames = []

        for item in serialized_images_data:
            
            filename = item['filename']
            serialized_image_data = item['data']

            # Decode base64 encoded image data
            decoded_image_data = base64.b64decode(serialized_image_data)

            # Convert decoded data to PIL image object
            image = Image.open(io.BytesIO(decoded_image_data))

            # Add the preprocessed image to the list
            preprocessed_image = preprocess_image(image)
            images.append(preprocessed_image)
            filenames.append(filename)


        images_array = np.array(images)
        predictions = model.predict(images_array)

        # Process predictions to get class names and probabilities
        results = []
        for i, prediction in enumerate(predictions):
            predicted_class_index = np.argmax(prediction)
            predicted_class_name = class_names.get(str(predicted_class_index), "Unknown Class")
            probability = prediction[predicted_class_index]
            results.append({
                'filename': filenames[i],
                'predicted_class': predicted_class_name,
                'probability': float(probability)
            })

        # Return results as JSON response
        return jsonify({'results': results}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/results', methods=['GET'])
def display_results():
    # Load the results from the JSON file
    if os.path.exists('results.json'):
        with open('results.json', 'r') as f:
            results = json.load(f)
    else:
        results = []

    # HTML template for displaying results
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image Classification Results</title>
    </head>
    <body>
        <h1>Image Classification Results</h1>
        <table border="1">
            <tr>
                <th>Image</th>
                <th>Filename</th>
                <th>Predicted Class</th>
                <th>Probability</th>
            </tr>
            {% for result in results %}
            <tr>
                <td><img src="data:image/jpeg;base64,{{ result.image_data }}" width="60" height="80"></td>
                <td>{{ result.filename }}</td>
                <td>{{ result.predicted_class }}</td>
                <td>{{ result.probability }}</td>
            </tr>
            {% endfor %}
        </table>
    </body>
    </html>
    """

    # Add encoded image data to results for display
    for result in results:
        with open(os.path.join('batches', result['filename']), 'rb') as f:
            image_data = f.read()
            encoded_image_data = base64.b64encode(image_data).decode('utf-8')
            result['image_data'] = encoded_image_data

    return render_template_string(html_template, results=results)


if __name__ == '__main__':
    # Start the Flask app
    app.run(host='127.0.0.1', port=5001)
