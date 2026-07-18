# Penguin Species Prediction API

## Description

This project predicts the species of a penguin using four numeric measurements.

The model is trained using the Palmer Penguins dataset and served using FastAPI. The application is containerized using Docker. vhvjhv

## Features

- FastAPI REST API
- Random Forest Classifier
- Dockerized application
- Swagger UI documentation

## Dataset

Input Features:
- bill_length_mm
- bill_depth_mm
- flipper_length_mm
- body_mass_g

Target:
- species

## Install

```bash
pip install -r requirements.txt
```

## Train Model

```bash
python train_model.py
```

## Run API

```bash
uvicorn main:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

## Docker Build

```bash
docker build -t penguin-fastapi:1.0 .
```

## Run Docker

```bash
docker run --rm --name penguin-api -p 8001:8000 penguin-fastapi:1.0
```

Open:

```
http://127.0.0.1:8001/docs
```
