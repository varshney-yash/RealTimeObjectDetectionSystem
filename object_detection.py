from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import YolosForObjectDetection, YolosImageProcessor

def load_image(image_path):
    return Image.open(image_path)

def get_model_and_processor(model_name="hustvl/yolos-tiny"):
    processor = YolosImageProcessor.from_pretrained(model_name)
    model = YolosForObjectDetection.from_pretrained(model_name)
    return processor, model

def process_image(image, processor, model):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    return processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]

def draw_results(image, results, font):
    draw = ImageDraw.Draw(image)
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if score > 0.7:
            box_values = box.tolist()
            label = model.config.id2label[label.item()]
            draw.rectangle(box_values, outline="red", width=5)
            draw.text(box_values[0:2], label, fill="red", font=font)

if __name__ == "__main__":
    root_directory = Path(__file__).resolve().parent.parent
    picture_path = root_directory / "RealTimeObjectDetectionSystem" / "assets" / "test-image3.jpg"
    font_path = root_directory / "RealTimeObjectDetectionSystem" / "assets" / "OpenSans-ExtraBold.ttf"

    image = load_image(picture_path)
    processor, model = get_model_and_processor()
    results = process_image(image, processor, model)
    font = ImageFont.truetype(str(font_path), 24)

    print("\n\nPrediction completed->", results, "\n\n")

    draw_results(image, results, font)
    image.show()
