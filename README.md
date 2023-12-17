# ML-Zoomcamp-Capstone-Project-2
This is the final capstone project for the ML Zoomcamp

# Project README: Predicting Data Scientist Salaries with Deep Learning

## Overview

This project leverages a sophisticated deep learning model to predict salaries for data scientists. Our approach combines a Deep Neural Network (DNN), Convolutional Neural Network (CNN), and Recurrent Neural Network (RNN) into a stacked architecture for enhanced accuracy and performance.

## How to Use the Project

To utilize this project, follow these sequential steps:

### Step 1: Run the Jupyter Notebook

Begin by executing the Jupyter Notebook included in the project. This notebook contains the necessary code to initialize and train the model. Ensure you have Jupyter installed and run the notebook by navigating to the project directory and executing:


**model.ipynb**


Navigate to the specific notebook file in the Jupyter interface and run all cells to train the model.

### Step 2: Run the `predict.py` Script Locally

After training the model, the next step is to use it for predictions. This is done through the `predict.py` script. Execute this script in your local environment:


**predict.py**


This script will use the trained model to make salary predictions based on the input data provided.

### Step 3: Running with Docker (Optional)

For those who prefer a Dockerized environment, follow these steps:

1. **Build the Docker Image:**

   Use the Dockerfile provided in the folder to build your Docker image. Run the following command in the terminal:

   ```shell
   docker build -t my-python-app .
   ```

2. **Run the Application in a Docker Container:**

   After building the image, run the application inside a Docker container with the following command:

   ```shell
   docker run my-python-app
   ```

   This will execute the `predict.py` script within a Docker container, utilizing the same model and environment settings defined in the Dockerfile.

## Additional Information

- **Data Preparation:** Before running the Jupyter Notebook, ensure your data is properly formatted and located in the designated directory as specified in the notebook.
- **Dependencies:** All necessary Python dependencies are listed in `requirements.txt`. Install them using `pip install -r requirements.txt`.

## Screenshots

![Screenshot from 2023-12-17 19-44-05](https://github.com/Bcopeland64/ML-Zoomcamp-Capstone-Project-2/assets/47774770/8f4abdbb-77fa-46d7-8836-1e3e8d3b3451)


