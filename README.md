# GLIP 實戰
## GLIP 簡介
GLIP: Grounded Language-Image Pre-training
https://github.com/microsoft/GLIP
一種用於學習對象級、語言感知和語義豐富的視覺表示的基礎語言圖像預訓練 (GLIP) 模型。**GLIP 統一了對象檢測和短語**基礎以進行預訓練。
統一帶來兩個好處：
1）它允許 GLIP 從檢測和基礎數據中學習，以改進這兩個任務並引導一個良好的基礎模型
2 ) GLIP 可以通過以自我訓練的方式生成接地框來利用大量的圖像文本對，從而使學習到的表示具有豐富的語義。
- 在我們的實驗中，我們在 27M 基礎數據上預訓練 GLIP，包括 3M 人工註釋和 24M 網絡抓取的圖像文本對。學習到的表示展示了對各種對象級識別任務的強大的零樣本和少樣本可遷移性。
- ![](https://github.com/microsoft/GLIP/raw/main/docs/lead.png)

- ## 模型使用
    ## 範例1
    想要框選旁邊正在加藥的桶子輸入 **bucket 水桶** ，但使用模型只認得小的水桶
    ![](https://i.imgur.com/lbKSJm6.jpg)

    利用用語義說明 **blue rags on big bucket** **藍色抹布在大水桶** 上成功框選到加藥桶
    ![](https://i.imgur.com/sDY7lxO.jpg)

    ## 範例2
    **John wearing white suit and Ben wearing black clothes** 模型沒有訓練過防護衣，但是可以使用**white suit 白色套裝**也能成功框選，且模型能夠根據語意推斷哪個是**John** 哪個是 **Ben**。
    - ![](https://i.imgur.com/kOxo8qT.jpg)


- ## Docker 建置環境
- ### 範例程式
	- 打開window終端機輸入
	- ```
	  docker run -it -v C:\Users\ZME\AI:/workspace -p 18888:8888 -p 15000:5000 --name GLIP_gpu --gpus all pengchuanzhang/pytorch:ubuntu20.04_torch1.9-cuda11.3-nccl2.9.9
	  ```
- ### 參數詳解
- `-it` 是 `--interactive` +的縮寫 `--tty` 。當您 `docker run` 使用此命令時，它會帶您直接進入容器
- `-v` 共用資料夾 `主機共用路徑` : `docker內共路徑`
- `-p` 是虛擬機內外port 轉發  `主機 port` : `docker port`
- `--name` Containers 名稱
- `--gpus all` docker 使用所有 gpu
-
- 依照Cuda版本選擇映像檔
	- Cuda 11.3 映像檔
		- `pengchuanzhang/pytorch:ubuntu20.04_torch1.9-cuda11.3-nccl2.9.9`
	- Cuda10.2 映像檔
		- `pengchuanzhang/maskrcnn:ubuntu18-py3.7-cuda10.2-pytorch1.9`
-

- ## Docker 環境啟動
### 1. 選擇 Containers
![](https://i.imgur.com/S4QvK6B.png)

- ### 2. **打開終端機** OPEN IN TERMINAL
- ![](https://i.imgur.com/cn7mHoz.png)

- ### 3. 輸入`jupyter lab`
- ### 4. 複製 `token`
	- *http://hostname:**8888**/token=`b2c967cb247f79da97803bd7b66d52ca527cf78b1c52974e``*
- ### 5. 瀏覽器輸入`
	- http://localhost:18888/ 進入 jupyter lab
- ### 6. 貼上 token 設定 jupyter lab 密碼
- ![](https://i.imgur.com/cHrjY7I.png)

- ## 建立Docker內程式環境
- ### 1. 打開Jupyter Lab 終端機
![](https://i.imgur.com/TcSaBzQ.png)

- ### 2. 輸入 `git clone https://github.com/microsoft/GLIP` 下載*GLIP github*
- ### 3. 進入 *GLIP* 資料夾 `cd GLIP`
- ### 4. 創建 Demo.ipynb
![](https://i.imgur.com/TKREC4J.png)

- ### 5. 安裝所需套件
- ```
  ! pip install einops shapely timm yacs tensorboardX ftfy prettytable pymongo
  ! pip install transformers 
  ! pip install ipywidgets
  ! python setup.py build develop --user
  ```
- ### 6. import  所需套件
- ```
  import matplotlib.pyplot as plt
  import matplotlib.pylab as pylab
  
  import requests
  from io import BytesIO
  from PIL import Image
  import numpy as np
  pylab.rcParams['figure.figsize'] = 20, 12
  from maskrcnn_benchmark.config import cfg
  from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
  
  def load(url):
      """
      Given an url of an image, downloads the image and
      returns a PIL image
      """
      response = requests.get(url)
      pil_image = Image.open(BytesIO(response.content)).convert("RGB")
      # convert to BGR format
      image = np.array(pil_image)[:, :, [2, 1, 0]]
      return image
  
  def imshow(img, caption):
    plt.imshow(img[:, :, [2, 1, 0]])
    plt.axis("off")
    plt.figtext(0.5, 0.09, caption, wrap=True, horizontalalignment='center', fontsize=20)
  ```
- ###  7. 下載模型權重
- Use this command for evaluate the GLPT-T model
- `weight_file` 檔案較大提供GD [載點](https://drive.google.com/file/d/1Q7DVX_dY1YcGeWYezrqZU7msOBdE7k6E/view?usp=sharing)
- 將`glip_tiny_model_o365_goldg_cc_sbu.pth` 放在 `MODEL` 目錄底下
- ```
  # ! wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_tiny_model_o365_goldg_cc_sbu.pth -O MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth
  config_file = "configs/pretrain/glip_Swin_T_O365_GoldG.yaml"
  weight_file = "MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth"
  ```
- update the config options with the config file
- manual override some options
- ```
  cfg.local_rank = 0
  cfg.num_gpus = 1
  cfg.merge_from_file(config_file)
  cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
  cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
  ```
- (第一次執行要下載資料比較久，notebook進度條有bug看不到會比較空虛)
- ```
  glip_demo = GLIPDemo(
      cfg,
      min_image_size=800,
      confidence_threshold=0.7,
      show_mask_heatmaps=False
  )
  ```
- 可以改使用`.py`執行可正常使用進度條 創建`download.py`
- ```
  from maskrcnn_benchmark.config import cfg
  from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
  
  
  if __name__ == '__main__':
  
      # Use this command for evaluate the GLPT-T model
      # ! wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_tiny_model_o365_goldg_cc_sbu.pth -O MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth
      config_file = "configs/pretrain/glip_Swin_T_O365_GoldG.yaml"
      weight_file = "MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth"
  
      # Use this command to evaluate the GLPT-L model
      # ! wget https://penzhanwu2bbs.blob.core.windows.net/data/GLIPv1_Open/models/glip_large_model.pth -O MODEL/glip_large_model.pth
      # config_file = "configs/pretrain/glip_Swin_L.yaml"
      # weight_file = "MODEL/glip_large_model.pth"
  
      # update the config options with the config file
      # manual override some options
      cfg.local_rank = 0
      cfg.num_gpus = 1
      cfg.merge_from_file(config_file)
      cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
      cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
      glip_demo = GLIPDemo(
          cfg,
          min_image_size=800,
          confidence_threshold=0.7,
          show_mask_heatmaps=False
      )
  ```
- ## Bug修正
- `GLIPDemo` class 在執行時會未定義 `self.color` 取代為 255  不影響使用
- ```
    glip_demo.color=255
  ```
- ## 讀取圖像方式
	- Internet 讀取圖像
		- `image = load('http://farm4.staticflickr.com/3693/9472793441_b7822c00de_z.jpg')`
	- PIL Imgae
		- `image = np.array(Image.open('./4.png'))[:,:,::-1]`
		- 1. 讀取圖片
		- 2. PIL格式轉為np.array
		- 3. `[:,:,::-1]` RGB -> BGR (PIL圖片顯示為RGB cv2為BGR)
	- open-cv
		- `cv2.imread('./9.png')`
- ## 成果展示 1
	- ### 模型檢測
		- 檢測 **a man,the cone,  hard hat **
		- 檢測 人、三角錐、安全帽
- ```
  image = np.array(Image.open('./4.png'))[:,:,::-1]
  caption = 'a man . the cone . hard hat' # the caption can also be the simple concatonation of some random categories.
  result, _ = glip_demo.run_on_web_image(image, caption, 0.5)
  imshow(result, caption)
  ```
![](https://i.imgur.com/rvN9eb7.jpg)

-
- ## 成果展示 2
	- 利用語意說明  **a man wearing hard hat and cone next to him**
	- 有個人戴著安全帽，三角錐在他身旁
- ```
  image = np.array(Image.open('./4.png'))[:,:,::-1]
  caption = 'a man wearing hard hat and cone next to him' # the caption can also be the simple concatonation of some random categories.
  result, _ = glip_demo.run_on_web_image(image, caption, 0.5)
  imshow(result, caption)
  ```
-
![](https://i.imgur.com/2ovmfSc.jpg)


- ## 輸出Pytorch Yolov5 標註檔
	- ### 將cv2 `左上` -> `右下` 框選方式改為 yolov5 格式 `中心點 長寬`
		- ```
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
		  ```
-
	- ### 模型`predictions` 篩選閥值，取得標籤，寫出`.txt`輸出位置與影像輸入相同
		- ```
		  def make_label(img_path,caption):
		      image = cv2.imread(img_path)
		      txt_path = os.path.splitext(img_path)[0]
		      predictions = glip_demo.compute_prediction(image, caption)
		      glip_demo.confidence_threshold = 0.5
		      top_predictions = glip_demo._post_process_fixed_thresh(predictions)
		      boxs = top_predictions.bbox
		      index = top_predictions.get_field("labels")
		      labels = [glip_demo.entities[i - 1]for i in index]
		      h,w,_ = image.shape
		      xywhs = xyxy2xywhn(x=boxs,w=w,h=h)
		      
		      with open(f'{txt_path}.txt', 'a') as f:
		          for cls,xywh in zip(index,xywhs):
		              line = (cls-1, *xywh)
		              f.write(('%g ' * len(line)).rstrip() % line + '\n')
		              
		      with open(os.path.join(''.join(os.path.split(txt_path)[:-1]),'labels.txt'), 'w') as f:
		          for c in glip_demo.entities:
		              f.write(c+ '\n')
		  ```
          
-
	- ### 測試 Function
	- ```
	  img_path = './pic/4.png'
	  caption = 'a man wearing hard hat and cone next to him' 
	  make_label(image,caption)
	  ```
-
	- ### 輸出結果 `./pic/4.txt`
	- ```
	  0 0.770998 0.547756 0.072838 0.212098
	  1 0.779665 0.513667 0.0603401 0.136979
	  2 0.936433 0.467944 0.034296 0.0650564
	  4 0.368129 0.365073 0.0397727 0.0945886
	  0 0.777506 0.514885 0.0635377 0.141048
	  
	  ```
      
# API 架設
### 程式測試完成利用 (本機)<-->(Docker) 的API，傳輸 (圖片+語意)<-->(類別+座標)
- 本機將圖片轉為base64字串+語意透過API傳輸
- 透過docker內的FastAPI回傳Json格式類別+座標
- ## 本機設置
    - ### 在建置container預留了 `-p 15000:5000` 給API連接
    - 設置API PORT 連至 `15000` 與 docker 內 `5000` 連接
    - `base64_str` 為 base64格式的圖片，型態為字串
    - `caption` 需要GLIP辨識的語意
    - ```
        import base64
        from PIL import Image
        from io import BytesIO
        import requests,json

        def api(img_base64):
            r = requests.post('http://127.0.0.1:15000/upload', 
            data = {
                'base64_str':f'{img_base64}',
                'caption': 'a man hard hat'
            })

            return json.loads(r.text)
      ```
    - 將PIL圖片轉為base64連同語意一起發送到API
        ```
            caption = 'cone . a man'
            image = Image.open(r'temp\outputs\3703(G2A_L10_1_D)GED1_2022_11_07_7_57_59_0000.png')
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue())
            api(str(img_base64)[2:-1])
        ```
    
- ## Docker設置
    ```
    pip install python-multipart
    pip install fastapi
    ```
    - ### 新增`main.py`
    ```
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
    ```
    
- ### 將收到的 base64 圖片轉為 PIL 格式
    ```
        def base64_to_image(base64_str, image_path=None):
            base64_data = re.sub('^data:image/.+;base64,', '', base64_str)
            byte_data = base64.b64decode(base64_data)
            image_data = BytesIO(byte_data)
            img = Image.open(image_data)
            if image_path:
                img.save(image_path)
            return img
    ```
- ### 坐標系轉換
    ```
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
    ```
    
- ### 預測結果打包成Json格式回傳  
	- {  
	  "0": { `圖片內第幾個ROI`  
	  "index": 0, `物體類別的index`  
	  "label": "a man", `物體類別`  
	  "x": 0.7348242402076721, `ROI中心點x座標/圖片寬度`  
	  "y": 0.7646925449371338, `ROI中心點y座標/圖片高度`  
	  "w": 0.16020514070987701, `ROI寬度/圖片寬度`  
	  "h": 0.2502952814102173 `ROI高度/圖片高度`  
	  },  
	  "1": {  
	  "index": 0,  
	  "label": "a man",  
	  "x": 0.7718421816825867,  
	  "y": 0.7066231966018677,  
	  "w": 0.04969634860754013,  
	  "h": 0.08035018295049667  
	  }  
	  }  
  ```
  def predict2json(image,caption):
      image = np.array(image)[:,:,::-1]
      predictions = glip_demo.compute_prediction(image, caption)
      glip_demo.confidence_threshold = 0.5
      top_predictions = glip_demo._post_process_fixed_thresh(predictions)
      boxs = top_predictions.bbox
      index = top_predictions.get_field("labels")
      h,w,_ = image.shape
      xywhs = xyxy2xywhn(x=boxs,w=w,h=h)

      res = {}
      for c, (i,loc) in enumerate(zip(index,xywhs)):
          x,y,w,h = loc
          res[c] = {}
          res[c]['index'] = int(i) -1
          res[c]['label'] = glip_demo.entities[int(i) -1]
          res[c]['x'] = float(x)
          res[c]['y'] = float(y)
          res[c]['w'] = float(w)
          res[c]['h'] = float(h)
      return res
  ```
      
- ### 啟動GLIP model   
  ```
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
  ```
-  
- ### 架設FastAPI  
	- uvicorn.run(app, host="0.0.0.0", port=`5000`) 可自由設定port  
  ```
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
  ```
