import os
import base64
import requests
import json
import schedule
import time


folder_path = 'batches'


def serialize():
 
    serialized_images = []

    # Iterate over each image file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            # Read image from file
            with open(os.path.join(folder_path, filename), 'rb') as f:
                image_data = f.read()

            # Encode image data as base64
            encoded_image_data = base64.b64encode(image_data)

            # Convert bytes to string (if necessary)
            encoded_image_string = encoded_image_data.decode('utf-8')

           
            serialized_images.append({'filename': filename, 'data': encoded_image_string})
    return serialized_images

def run_batch_prediction():

    serialized_images = serialize()
    # Define the URL of the server
    url = 'http://127.0.0.1:5001/predict_batch'

   
    payload = {'images': serialized_images}

    # Send the POST request to the server
    response = requests.post(url, json=payload)
    # Parse the JSON response
    response_data = response.json()

    # Print the response from the server
    results = []
    if 'results' in response_data:
        for result in response_data['results']:
            filename = result['filename']
            predicted_class = result['predicted_class']
            probability = result['probability']
            results.append({
                    'filename': filename,
                    'predicted_class': predicted_class,
                    'probability': probability
            })
            print(f"Object: {filename} Class: {predicted_class} Probability: {probability:.4f}")
        else:
            print("Error:", response_data.get('error', 'Unknown error'))

        # Save the results to a JSON file
        with open('results.json', 'w') as f:
            json.dump(results, f)


# Schedule the batch prediction to run every night at 1 AM
schedule.every().day.at("15:43").do(run_batch_prediction)

# Keep the script running
while True:
    schedule.run_pending()
    time.sleep(1)