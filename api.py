import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from pydantic import BaseModel
from transformers import YolosForObjectDetection, YolosImageProcessor

class Object(BaseModel):
    box: list[float]
    label: str

class Objects(BaseModel):
    objects: list[Object]

class ObjectDetection:
    def __init__(self):
        self.image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
        self.model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny")

    def predict(self, image: Image.Image) -> Objects:
        if not self.image_processor or not self.model:
            raise RuntimeError("Model is not loaded")
        
        inputs = self.image_processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = self.image_processor.post_process_object_detection(
            outputs, threshold=0.7, target_sizes=target_sizes
        )[0]

        objects = [Object(box=box.tolist(), label=self.model.config.id2label[label.item()])
                   for score, label, box in zip(results["scores"], results["labels"], results["boxes"]) if score > 0.7]
        return Objects(objects=objects)

app = FastAPI()

@app.on_event("startup")
async def load_model():
    global object_detection
    object_detection = ObjectDetection()

@app.post("/object-detection", response_model=Objects)
async def post_object_detection(image_file: UploadFile = File(...)) -> Objects:
    try:
        with Image.open(image_file.file).convert("RGB") as image_object:
            return object_detection.predict(image_object)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

