# RealTimeObjectDetectionSystem

FastAPI websockets and huggingface

To run locally, make an virtual env, install requirements and run server with uvicorn app:app

## An object detection system that works in real time with video input

The browser will send a stream of images into the WebSocket from the webcam. The application will run an object detection algorithm and send back the coordinates and label of each detected object in the image.
