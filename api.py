from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from io import BytesIO
from PIL import Image
import sys
import os
import base64

# Add path to use code from diss-img-tool-lw/
sys.path.append(os.path.join(os.path.dirname(__file__), 'diss-img-tool-lw'))

from models import create_model
from Tree_Image_Generator import generate_images
from Tree_Image_Generator_opts import bicycle_gan_opts

app = FastAPI()

# Load model once on startup
opt = bicycle_gan_opts('high_ren')
model = create_model(opt)
model.setup(opt)

# POST endpoint
@app.post("/generate/")
async def generate(sketch: UploadFile = File(...), style: UploadFile = File(...)):
    sketch_image = Image.open(BytesIO(await sketch.read()))
    style_image = Image.open(BytesIO(await style.read()))

    outputs, _ = generate_images(model, sketch_image, style_image)

    print(outputs)
    def image_to_base64(img):
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    return {
        "images": [image_to_base64(img) for img in outputs]
    }
