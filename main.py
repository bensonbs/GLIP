import re
import os
import torch
import base64
import uvicorn
import numpy as np

from io import BytesIO
from PIL import Image
from typing import Union
from fastapi import FastAPI, File, Form
from pydantic import BaseModel

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo

def base64_to_image(base64_str, image_path=None):
    base64_data = re.sub('^data:image/.+;base64,', '', base64_str)
    byte_data = base64.b64decode(base64_data)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data)
    if image_path:
        img.save(image_path)
    return img

def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y

def predict2json(image,caption):
    image = np.array(image)[:,:,::-1]
    predictions = glip_demo.compute_prediction(image, caption)
    glip_demo.confidence_threshold = 0.5
    top_predictions = glip_demo._post_process_fixed_thresh(predictions)
    boxs = top_predictions.bbox
    index = top_predictions.get_field("labels")
    probs  = top_predictions.get_field("scores")
    h,w,_ = image.shape
    xywhs = xyxy2xywhn(x=boxs,w=w,h=h)
    
    res = {}
    for c, (i,loc,prob) in enumerate(zip(index,xywhs,probs)):
        x,y,w,h = loc
        res[c] = {}
        res[c]['index'] = int(i) -1
        res[c]['label'] = glip_demo.entities[int(i) -1]
        res[c]['prob'] = float(prob)
        res[c]['x'] = float(x)
        res[c]['y'] = float(y)
        res[c]['w'] = float(w)
        res[c]['h'] = float(h)
    return res

    
    
config_file = "configs/pretrain/glip_Swin_T_O365_GoldG.yaml"
weight_file = "MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth"

cfg.local_rank = 0
cfg.num_gpus = 1
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

glip_demo = GLIPDemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.5,
    show_mask_heatmaps=False
)

app = FastAPI()


class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/upload")
def upload(base64_str: str = Form(...), caption: str = Form(...)):
    try:
        image = base64_to_image(base64_str)
        res = predict2json(image,caption)
    except Exception as e:
        return {"message": f"{e}"}

    return res

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)