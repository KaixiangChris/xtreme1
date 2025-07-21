# YOLOv8 ImageNet Integration for Xtreme1

This directory contains the YOLOv8 ImageNet detection model service that integrates with the Xtreme1 platform to provide AI-powered object detection using ImageNet classes.

## Overview

The YOLOv8 ImageNet model service provides:
- Object detection using YOLOv8 architecture
- Support for 1000 ImageNet classes (animals, vehicles, objects, etc.)
- REST API compatible with Xtreme1's model interface
- Configurable confidence thresholds and class filtering

## Architecture

```
┌─────────────────┐    HTTP/JSON    ┌──────────────────┐
│   Xtreme1       │ ──────────────► │  YOLOv8 Service  │
│   Backend       │                 │                  │
│                 │ ◄────────────── │  FastAPI + YOLO  │
└─────────────────┘    Detection    └──────────────────┘
                       Results
```

## Service Components

### 1. FastAPI Application (`app.py`)
- HTTP server exposing detection endpoints
- Image download and preprocessing
- YOLOv8 model inference
- Results formatting for Xtreme1

### 2. ImageNet Classes (`imagenet_classes.py`)
- Mapping of class IDs to human-readable names
- Representative subset of 1000 ImageNet classes
- Easily extensible for additional classes

### 3. Docker Container
- Python 3.9 runtime
- YOLOv8 dependencies (ultralytics, torch, etc.)
- Automatic model weight download
- Health checking

## API Endpoints

### POST `/image/recognition`
Detect objects in images using YOLOv8.

**Request Format:**
```json
{
  "datas": [
    {
      "id": 12345,
      "url": "https://example.com/image.jpg"
    }
  ]
}
```

**Response Format:**
```json
{
  "code": 200,
  "message": "success",
  "data": [
    {
      "id": 12345,
      "confidence": 0.85,
      "objects": [
        {
          "classId": "285",
          "className": "egyptian_cat",
          "confidence": 0.92,
          "boundingBox": {
            "x": 100.5,
            "y": 200.3,
            "width": 150.2,
            "height": 180.7
          }
        }
      ]
    }
  ]
}
```

### GET `/health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## Installation & Deployment

### 1. Building the Service

```bash
# Navigate to the service directory
cd models/yolov8-imagenet

# Build Docker image
docker build -t yolov8-imagenet .
```

### 2. Running with Xtreme1

The service is automatically included when running Xtreme1 with the `model` profile:

```bash
# Start Xtreme1 with model services
docker compose --profile model up -d

# Check service status
docker compose ps
```

The YOLOv8 service will be available at: `http://localhost:8296`

### 3. Manual Testing

```bash
# Test health endpoint
curl http://localhost:8296/health

# Test detection with a sample image
curl -X POST http://localhost:8296/image/recognition \
  -H "Content-Type: application/json" \
  -d '{
    "datas": [
      {
        "id": 1,
        "url": "https://example.com/cat.jpg"
      }
    ]
  }'
```

## Configuration

### Model Selection
By default, the service uses `yolov8n.pt` (nano model). You can change this in `app.py`:

```python
# Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
model = YOLO('yolov8s.pt')  # Use small model for better accuracy
```

### GPU Support
To enable GPU acceleration, uncomment the runtime line in `docker-compose.yml`:

```yaml
yolov8-imagenet:
  build: ./models/yolov8-imagenet
  runtime: nvidia  # Uncomment this line
```

### Class Customization
To add or modify ImageNet classes, edit `imagenet_classes.py`:

```python
IMAGENET_CLASSES = {
    # Add new class mappings
    1000: "custom_object",
    # ... existing classes
}
```

## Integration with Xtreme1

### Backend Integration
The service integrates with Xtreme1 through:

1. **Model Handler**: `YoloV8ImageNetModelHandler.java`
2. **Request/Response Converters**: `YoloV8RequestConverter.java`, `YoloV8ResponseConverter.java`
3. **HTTP Client**: `YoloV8ModelHttpCaller.java`
4. **Database Model**: Model ID 3 with ImageNet classes

### Database Configuration
The YOLOv8 model is pre-configured in the database with:
- Model ID: 3
- Model Code: `YOLOV8_IMAGENET`
- URL: `http://yolov8-imagenet:5000/image/recognition`
- Representative ImageNet classes (40+ examples)

### Frontend Integration
The model appears in the Xtreme1 UI as "YOLOv8 ImageNet Detection" and can be used for:
- Image annotation assistance
- Batch processing
- Model performance evaluation

## Troubleshooting

### Common Issues

1. **Model Loading Failed**
   ```bash
   # Check if model weights are downloaded
   docker logs yolov8-imagenet
   ```

2. **Out of Memory**
   ```bash
   # Use smaller model variant
   # Change yolov8n.pt to yolov8n.pt in app.py
   ```

3. **Slow Detection**
   ```bash
   # Enable GPU support or use faster model
   # Add runtime: nvidia to docker-compose.yml
   ```

4. **Connection Issues**
   ```bash
   # Verify service is running
   docker compose ps yolov8-imagenet
   
   # Check service logs
   docker compose logs yolov8-imagenet
   ```

### Performance Optimization

1. **Model Size**: Choose appropriate model variant
   - `yolov8n.pt`: Fastest, lower accuracy
   - `yolov8s.pt`: Balanced speed/accuracy
   - `yolov8m.pt`: Better accuracy, slower
   - `yolov8l.pt`, `yolov8x.pt`: Best accuracy, slowest

2. **Batch Processing**: Process multiple images in single request

3. **GPU Acceleration**: Use NVIDIA runtime for faster inference

## Development

### Adding New Classes
1. Update `imagenet_classes.py` with new class mappings
2. Add corresponding database entries in `V2__Init_data.sql`
3. Rebuild the service

### Custom Models
To use custom YOLOv8 models:
1. Replace model loading in `app.py`
2. Update class mappings accordingly
3. Adjust confidence thresholds if needed

## License

This integration follows the same license as the main Xtreme1 project (Apache 2.0).

## Support

For issues and questions:
1. Check the main Xtreme1 documentation
2. Review Docker logs for service issues
3. Test API endpoints directly for debugging 