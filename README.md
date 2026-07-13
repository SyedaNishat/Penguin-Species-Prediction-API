# Penguin Species Prediction API

## Overview
This project predicts penguin species using a Random Forest Classifier trained on the Palmer Penguins dataset. The trained model is served through a FastAPI REST API and deployed using Docker.

## Technologies
- Python
- Pandas
- Scikit-learn
- FastAPI
- Docker
- Joblib

## Features
- Data preprocessing
- Model training
- REST API for predictions
- Swagger documentation
- Dockerized deployment

## Run Locally

pip install -r requirements.txt

python train_model.py

uvicorn main:app --reload

## Docker

docker build -t penguin-fastapi:1.0 .

docker run --rm -p 8001:8000 penguin-fastapi:1.0
