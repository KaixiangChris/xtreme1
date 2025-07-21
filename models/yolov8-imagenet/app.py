from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import logging
from ultralytics import YOLO
import requests
from PIL import Image
import io
import numpy as np
from imagenet_classes import IMAGENET_CLASSES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="YOLOv8 ImageNet Detection Service", version="1.0.0")

# Initialize YOLOv8 model
model = None

def load_model():
    """Load YOLOv8 model"""
    global model
    try:
        model = YOLO('yolov8n.pt')  # You can change to yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
        logger.info("YOLOv8 model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load YOLOv8 model: {e}")
        raise e

# Pydantic models for API
class ImageData(BaseModel):
    id: int
    url: str

class DetectionRequest(BaseModel):
    datas: List[ImageData]

class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float

class DetectionObject(BaseModel):
    classId: str
    className: str
    confidence: float
    boundingBox: BoundingBox

class DetectionResponse(BaseModel):
    id: int
    confidence: Optional[float] = None
    objects: List[DetectionObject]

class ApiResponse(BaseModel):
    code: int
    message: str
    data: List[DetectionResponse]

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/image/recognition", response_model=ApiResponse)
async def detect_objects(request: DetectionRequest):
    """
    Detect objects in images using YOLOv8
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        results = []
        
        for image_data in request.datas:
            logger.info(f"Processing image ID: {image_data.id}, URL: {image_data.url}")
            
            # Download image from URL
            try:
                response = requests.get(image_data.url, timeout=30)
                response.raise_for_status()
                image = Image.open(io.BytesIO(response.content))
                
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                    
            except Exception as e:
                logger.error(f"Failed to download/process image {image_data.id}: {e}")
                # Return empty result for failed images
                results.append(DetectionResponse(
                    id=image_data.id,
                    confidence=0.0,
                    objects=[]
                ))
                continue
            
            # Run YOLOv8 detection
            try:
                detections = model(image, verbose=False)
                detection_objects = []
                confidences = []
                
                # Process each detection
                for detection in detections:
                    boxes = detection.boxes
                    if boxes is not None and len(boxes) > 0:
                        for i in range(len(boxes)):
                            # Get bounding box coordinates (xyxy format)
                            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                            confidence = float(boxes.conf[i].cpu().numpy())
                            class_id = int(boxes.cls[i].cpu().numpy())
                            
                            # Convert to width/height format
                            width = x2 - x1
                            height = y2 - y1
                            
                            # Get class name from ImageNet classes
                            class_name = IMAGENET_CLASSES.get(class_id, f"class_{class_id}")
                            
                            # Create detection object
                            detection_obj = DetectionObject(
                                classId=str(class_id),
                                className=class_name,
                                confidence=confidence,
                                boundingBox=BoundingBox(
                                    x=float(x1),
                                    y=float(y1),
                                    width=float(width),
                                    height=float(height)
                                )
                            )
                            
                            detection_objects.append(detection_obj)
                            confidences.append(confidence)
                
                # Calculate average confidence
                avg_confidence = np.mean(confidences) if confidences else 0.0
                
                results.append(DetectionResponse(
                    id=image_data.id,
                    confidence=avg_confidence,
                    objects=detection_objects
                ))
                
                logger.info(f"Detected {len(detection_objects)} objects in image {image_data.id}")
                
            except Exception as e:
                logger.error(f"Detection failed for image {image_data.id}: {e}")
                results.append(DetectionResponse(
                    id=image_data.id,
                    confidence=0.0,
                    objects=[]
                ))
        
        return ApiResponse(
            code=200,
            message="success",
            data=results
        )
        
    except Exception as e:
        logger.error(f"Detection request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000) 