from fastapi import FastAPI, UploadFile, Response
from .inference import NougatBase
import logging

app = FastAPI(title="Latex Code OCR API", description="LaTex formula images from the clipboard into LaTex code using Transformers")

nougat_obj = NougatBase()

@app.post("/upload_latex_image/")
async def upload_file(file: UploadFile):
    image = await file.read()
    sequence = nougat_obj.inference_latex_code(image=image)
    logging.info(f"Sequence: {sequence}")    
    return Response(content=sequence, media_type="text/plain")
