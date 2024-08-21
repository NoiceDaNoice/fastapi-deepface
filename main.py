from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
from typing import Union

from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, Form, UploadFile
from datetime import datetime
import os

# uvicorn main:app --reload  --host localhost --port 8000     
# uvicorn main:app --reload  --host 10.251.2.119 --port 8000    
# uvicorn main:app --reload  --host 0.0.0.0 --port 8000 (ini semua ip bisa harusnya karena dynamic-ipung)
models = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
  "GhostFaceNet",
]

def verify_faces(img1_path_data, img2_path_data):
    result = DeepFace.verify(
        img1_path=img1_path_data,
        img2_path=img2_path_data,
        model_name= models[2]
    )
    return result

def detech_face(img):
    try:
        result = DeepFace.analyze(
            img_path=img
        )
        return result
    except:
        return "no face"


timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

def generate_file_location(filename: str) -> str:
    filename_without_extension = os.path.splitext(filename)[0]
    new_filename = f"{filename_without_extension}_{timestamp}"
    return f"upload/{new_filename}.jpg"

def generate_file_location_upload(filename: str) -> str:
    filename_without_extension = os.path.splitext(filename)[0]
    new_filename = f"{filename_without_extension}"
    return f"img/{new_filename}.jpg"

app = FastAPI()

@app.get("/")
def read_root():
    return {"result":"success"}

@app.post("/recognise")
async def recognise(file: UploadFile = File(...), img_name: str = Form(...)):
    file_location = generate_file_location(file.filename)
    try:
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())
            print(file_location)
        data = verify_faces(file_location, f"img/{img_name}")
        return JSONResponse(content=data, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        if os.path.exists(file_location):
            os.remove(file_location)
 
@app.post("/upload")
async def upload(file: UploadFile = File(...), img_name: str = Form(...)):
    file_location = generate_file_location_upload(img_name)
    try:
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())
            print(f"File saved at: {file_location}")
        detech = detech_face(file_location)
        if detech != "no face":
            return JSONResponse(content={"status": "success", "file_name": img_name}, status_code=200)
        else:
            if os.path.exists(file_location):
                os.remove(file_location)
            return JSONResponse(content={"status": "failed", "reason": detech}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
@app.get("/images")
async def list_images():
    try:
        files = os.listdir("img")
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}
        image_files = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]
        
        return JSONResponse(content={"images": image_files})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
