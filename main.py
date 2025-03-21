import cv2
import numpy as np
import pytesseract
import requests
import asyncio

from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO
from io import BytesIO
import os
from pathlib import Path
from PIL import Image
import uvicorn

app = FastAPI()

venv_path = Path(os.environ["VIRTUAL_ENV"])  # Get the venv path
pytesseract.pytesseract.tesseract_cmd = str(venv_path / "Scripts" / "tesseract.exe")

# ✅ Load YOLO Model (Force CPU Mode)
YOLO_MODEL_PATH = "best.pt"
model = YOLO(YOLO_MODEL_PATH).to("cpu")  # Force CPU Mode

def process_image(contents):
    """Process image for YOLO + OCR"""
    try:
        img = Image.open(BytesIO(contents))
        img = img.convert("RGB")  # Ensure correct format
        img = img.resize((640, 640))  # ✅ Resize for consistent YOLO input
        img_array = np.array(img)

        if img_array is None:
            raise ValueError("Image conversion failed!")

        # ✅ YOLO Detection (Force CPU)
        results = model(img_array, device="cpu")  
        detected_names = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]

                if confidence > 0.05:  # ✅ Lower threshold for weak detections
                    roi = img_array[y1:y2, x1:x2]

                    # ✅ Convert to grayscale for better OCR accuracy
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

                    text = pytesseract.image_to_string(roi_gray, config='--oem 3 --psm 6').strip()

                    if text:
                        detected_names.append(text)

                    # ✅ Draw bounding box for debugging
                    cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_array, f"{confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return {"detected_names": detected_names}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/detect-medicine/")
async def detect_medicine(image: UploadFile = File(...)):
    try:
        if not image:
            raise HTTPException(status_code=400, detail="No file uploaded")

        # Read and process image
        contents = await image.read()
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, process_image, contents)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=4, log_level="info")
