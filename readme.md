# Image Classification as a Service

This is a demonstration of what an image classification service using mlflow and flask can look like. 

## split.py 
splits the dataset into training, validation and testing sets. 

## model.py 
trains the model and saves it's properties as h5-file

## app.py

Flask app, where you can sent POST requests to


## requests.py

little script that serializes the batch data and sends requests. 
