from deepface import DeepFace
import os

class FaceRecognize(DeepFace):
    
    def __init__(self, mModel: list, debug: bool):
        self.default_model = [
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
        self.mModel = mModel
        self.debug = debug
        self.valid_formats = {'jpg', 'jpeg', 'png'}  # Gunakan set untuk pencarian lebih cepat
        
    def add_format(self, ext: str):
        self.valid_formats.add(ext)  # Gunakan set untuk menambahkan format baru
    
    def _check_image(self, img: str):
        """Fungsi internal untuk memeriksa keberadaan file dan formatnya."""
        if not os.path.exists(img):
            raise FileNotFoundError(f"Gambar tidak ditemukan: {img}")
        
        if img.split('.')[-1].lower() not in self.valid_formats:
            raise ValueError("Format gambar tidak valid. Harap gunakan jpg, jpeg, atau png.")
    
    def detect(self, img: str):
        self._check_image(img)
        
        if self.debug:
            print(f"Start detecting face on {img}")
        
        try:
            return True, super().analyze(img_path=img)
        except Exception as err:
            raise ValueError(err)
    
    def verify_face(self, img1_path_data, img2_path_data, model_name):
        if not model_name:
            raise ValueError(f"Pilih salah satu model dari berikut: \n{self.mModel}")
        
        if self.debug:
            print(f"Verify {img1_path_data} | {img2_path_data} . . .")
        
        try:
            return super().verify(img1_path=img1_path_data, img2_path=img2_path_data, model_name=model_name)
        except Exception as err:
            raise ValueError(err)
