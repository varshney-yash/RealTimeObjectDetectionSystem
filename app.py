import asyncio
import contextlib
import io
import logging
from pathlib import Path

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from transformers import YolosForObjectDetection, YolosImageProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Object(BaseModel):
    box: tuple[float, float, float, float]
    label: str

class Objects(BaseModel):
    objects: list[Object]

class ObjectDetection:
    image_processor: YolosImageProcessor | None = None
    model: YolosForObjectDetection | None = None

    def load_model(self) -> None:
        """Loads the model"""
        try:
            self.image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
            self.model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny")
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def predict(self, image: Image.Image) -> Objects:
        """Runs a prediction"""
        if not self.image_processor or not self.model:
            logger.error("Model is not loaded")
            raise RuntimeError("Model is not loaded")
        try:
            inputs = self.image_processor(images=image, return_tensors="pt")
            outputs = self.model(**inputs)

            target_sizes = torch.tensor([image.size[::-1]])
            results = self.image_processor.post_process_object_detection(
                outputs, target_sizes=target_sizes
            )[0]

            objects: list[Object] = []
            for score, label, box in zip(
                results["scores"], results["labels"], results["boxes"]
            ):
                if score > 0.7:
                    box_values = box.tolist()
                    label = self.model.config.id2label[label.item()]
                    objects.append(Object(box=box_values, label=label))
            return Objects(objects=objects)
        except Exception as e:
            logger.error(f"Failed to predict: {e}")
            raise

object_detection = ObjectDetection()

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    object_detection.load_model()
    yield

app = FastAPI(lifespan=lifespan)

async def receive(websocket: WebSocket, queue: asyncio.Queue):
    while True:
        try:
            bytes = await websocket.receive_bytes()
            queue.put_nowait(bytes)
        except asyncio.QueueFull:
            logger.warning("Queue is full, dropping frame")
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
            break
        except Exception as e:
            logger.error(f"Error receiving bytes: {e}")

async def detect(websocket: WebSocket, queue: asyncio.Queue):
    while True:
        try:
            bytes = await queue.get()
            image = Image.open(io.BytesIO(bytes))
            objects = object_detection.predict(image)
            await websocket.send_json(objects.model_dump())
        except Exception as e:
            logger.error(f"Error processing detection: {e}")

@app.websocket("/object-detection")
async def ws_object_detection(websocket: WebSocket):
    await websocket.accept()
    queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    receive_task = asyncio.create_task(receive(websocket, queue))
    detect_task = asyncio.create_task(detect(websocket, queue))
    try:
        done, pending = await asyncio.wait(
            {receive_task, detect_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        for task in done:
            task.result()
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "index.html")

static_files_app = StaticFiles(directory=Path(__file__).parent / "assets")
app.mount("/assets", static_files_app)