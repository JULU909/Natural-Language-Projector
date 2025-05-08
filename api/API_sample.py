import base64
import requests
import json
from PIL import Image
import io
import cv2
import numpy as np
import os
import sys

# Get the absolute path of the directory where this script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Move up one level to reach `Cleaned_Version`
base_dir = os.path.dirname(current_dir)

# Construct the path to Hi-sam
hisam_path = os.path.join(base_dir, "Hi-sam")



# Pythonでネットワーク内のPC等の機器から描画を行うためのプログラムサンプル
# 画像をそのまま表示させる場合は、image_Post_sampleを参考にしてください
# スポット描画の機能を使う場合は、draw_Spot_sampleを参考にしてください

# IPを入力
IP = '192.168.2.138' #使用している器具のIPを記入
# IP = 'localhost'

# JSONデータをマイクロLEDサーバーにPOSTする関数
def post2microAPP(ipaddr, jsondata):

    # HTTP POSTリクエストの送信
    url = 'http://' + ipaddr + ':8080'

    # POSTリクエストの送信
    response = requests.post(url, json=jsondata)

    # レスポンスの表示
    print(response.text)


# 画像表示API：画像データを読み込んで、マイクロLEDに送信し全画面表示する
def image_Post_sample(ipaddr):

    # 画像ファイルのパス
    image_path = './data/grid.bmp'

    # 画像の読み込み
    image = Image.open(image_path)

    # 画像をBase64形式にエンコード
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # POSTするデータ
    data = {
        "BASE64_IMG": image_base64
    }

    # Http-POST
    post2microAPP(ipaddr, data)

def send_image(ipaddr,image_path):

    image = Image.open(image_path)

    # 画像をBase64形式にエンコード
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # POSTするデータ
    data = {
        "BASE64_IMG": image_base64
    }

    # Http-POST
    post2microAPP(ipaddr, data)


# アニメーションAPIでローカルJSONファイルを読み込んで送信するサンプル
def send_Animation_localdata(ipaddr):

    jsonfile = './data/0_splash_newAPI.json'

    with open(jsonfile) as f:
        jsondata = json.load(f)
        # POSTするデータ
        data = {
            'SPOT_ANIMATION': jsondata
        }
        # Http-POST
        post2microAPP(ipaddr, data)

# アニメーションデータを作成し送信するサンプル
def send_Animation_data(ipaddr):

    # 1フレーム目のデータを生成
    data1 = {
        "time": 0,
        "SPOT_PARAMS": [
            {"ID":1, "STATE":True, "POSX":0, "POSY":500, "BRI":100, "SIZ":10}, # 左端にスポットが一つ
            {"ID":2, "STATE":False}, # 残りは消しておく
            {"ID":3, "STATE":False},
            {"ID":4, "STATE":False},
            {"ID":5, "STATE":False},
            {"ID":6, "STATE":False},
            {"ID":7, "STATE":False},
            {"ID":8, "STATE":False},
            {"ID":9, "STATE":False},
            {"ID":10, "STATE":False},
            {"ID":11, "STATE":False},
            {"ID":12, "STATE":False},
            {"ID":13, "STATE":False},
            {"ID":14, "STATE":False},
            {"ID":15, "STATE":False},
            {"ID":16, "STATE":False},
            {"ID":17, "STATE":False},
            {"ID":18, "STATE":False},
            {"ID":19, "STATE":False},
            {"ID":20, "STATE":False},
        ]

    }

    # 2フレームのデータを生成
    data2 = {
        "time": 5000, # 5秒後
        "SPOT_PARAMS": [
            {"ID":1, "POSX":1000} # スポットを右端に移動
        ]

    }

    # アニメーションデータを生成
    animation_data = {}
    animation_data["maxAnimationTime"] = 5000 # 5秒
    animation_data["frames"] = [
        data1,
        data2
    ]
    animation_data["repeat"] = True

    print(animation_data)

    # Http-POST
    post2microAPP(ipaddr, {"SPOT_ANIMATION": animation_data})

#台形補正のサンプル
def warpAnimation(ipaddr):

    default_value = [[0,0],[1023,0],[1023,1023],[0,1023]]
    value = [[100,0],[923,0],[1023,1023],[0,1023]]

    data = {}
    data['KST'] = value

    post2microAPP(ipaddr,data)

# スポット照明を制御するサンプル
def draw_Spot_sample(ipaddr):

    data = {}

    spot_params = []

    # スポット10個のデータを生成
    for id in range(1, 11):
        spotdata = {}
        spotdata["ID"] = id # スポットライトの番号
        spotdata["STATE"] = True # スポットライトの状態
        spotdata["SIZ"] = 8 # スポットライトのサイズ
        spotdata["POSX"] = id * 90 # スポットライトのX座標
        spotdata["POSY"] = 500 # スポットライトのY座標
        spotdata["COR"] = id * 10 # スポットライトの角丸め
        spotdata["BRI"] = id * 10 # スポットライトの明るさ
        spotdata["BLR"] = id * 10 # スポットライトのぼけ具合

        spot_params.append(spotdata)

    data["SPOT_PARAMS"] = spot_params

    post2microAPP(ipaddr, data)

##################################################################################################################################################################
# Add 2025/01/23
# 1. All off api is implemented ('ALL-OFF')
# 2. The maximum spot number has been increased from 10 to 20.
# 3. Images and videos are specified by Filename, not by their index number in APIs.
# 4. Defined Two types of contents fit : 'Canvas Fit' and 'Spot Fit'. 
#    By specifying either IMG_FIT_TYPE or MOV_FIT_TYPE in the API, you can choose whether the video fits the entire canvas or fits the size of the spot.
#    0: Canvas-Fit, 1: Spot-Fit(default)
# 5. Text api is implemented. Currently, there is only two panasonic licensed fonts: PUDGoDp2019R/Mt.ttf.
##################################################################################################################################################################

# 全消灯のAPI
def post_all_off(ipaddr):
    data = {'ALL_OFF': True}
    post2microAPP(ipaddr, data)

# 画像を登録して再生するサンプル
def embed_image_into_spot_sample(ipaddr):

    # 消灯しておく
    post_all_off(ipaddr)

    # 白い三角形を描画
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    points = np.array([[15, 35], [15, 15], [35, 35]], np.int32)
    cv2.fillPoly(image, [points], (255,255,255))

    # 画像をPNG形式でエンコード
    _, buffer = cv2.imencode('.png', image)

    # BASE64にエンコード
    base64data = base64.b64encode(buffer).decode('utf-8')

    # 画像の名称
    image_name = 'triangle.png'

    # 登録(画像は最大500枚まで保存できる)
    data1 = {
        'ADD_IMG':
        {
            'FILE_NAME': image_name,
            'IMG_DATA': base64data
        }
    }

    # POST
    post2microAPP(ipaddr, data1)

    # スポットとして使う
    data2 = {'SPOT_PARAMS': [
        {'ID': 1, 
         'STATE': True,
         'POSX': 500,
         'POSY': 500,
         'SIZ': 10,
         'BRI': 100,
         'COR': 0,
         'BLR': 0,
         'IMG': image_name, # コンテンツのファイル名を指定
         'IMG_FIT_TYPE': 1, # 0: Canvas-Fit, 1: Spot-Fit
         'MOV': 'default' 
        }]}
    
    # スポット描画
    post2microAPP(ipaddr, data2)

# mp4ファイルを読み込んで送信する
def embed_mp4file_into_spot_sample(ipaddr):

    # 消灯しておく
    post_all_off(ipaddr)

    file_path = './data/ameba3.mp4'

    # ファイルを開いてBase64エンコードする
    with open(file_path, 'rb') as mp4_file:
        mp4_data = mp4_file.read()
        mp4_data_base64 = base64.b64encode(mp4_data).decode('utf-8')

    # JSONオブジェクトを作成
    data1 = {
        'ADD_MOVIE': {
            'FILE_NAME': file_path.split('/')[-1],  # ファイル名を取得
            'MOVIE_DATA': mp4_data_base64,  # Base64エンコードされたデータ
        }
    }

    # POST
    post2microAPP(ipaddr, data1)

    movie_name = os.path.basename(file_path)

    # スポットとして使う
    data2 = {'SPOT_PARAMS': [
        {'ID': 1, 
         'STATE': True,
         'POSX': 500,
         'POSY': 500,
         'SIZ': 10,
         'BRI': 100,
         'COR': 0,
         'BLR': 0,
         'MOV': movie_name, # コンテンツのファイル名を指定
         'MOV_FIT_TYPE': 1, # 0: Canvas-Fit, 1: Spot-Fit
         'IMG': 'default'
        }]}
    
    # スポット描画
    post2microAPP(ipaddr, data2)

# draw
def draw_Text_sample(ipaddr):

    post_all_off(ipaddr)


    data = {
        "DRAW_TEXT": [
            {"TEXT": "Hello", "POSITION": [0,0], "BRIGHTNESS": 100, "SIZE": 12, "TYPE": "bold"}, 
            {"TEXT": "APIサンプル", "POSITION": [50,20], "BRIGHTNESS": 100, "SIZE": 32, "CENTERING": True}
        ]
    }

    # テキスト描画
    post2microAPP(ipaddr, data)


def image_path_sample(ipaddr,image_path):

    image = Image.open(image_path)

    # 画像をBase64形式にエンコード
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # POSTするデータ
    data = {
        "BASE64_IMG": image_base64
    }

    # Http-POST
    post2microAPP(ipaddr, data)



def send_image_sweep(ipaddr,image_path,speed=1,brightness=1,window_width=10):
    # params : 
    # speed 
    # Width of window
    # Brightness
    file_path = text_sweep(image_path,"/home/jazz/Harish_ws/Demo/microled/agents/temp_storage",speed= speed,brightness=brightness,window_width = window_width)
    print("file path is in " , file_path)
    # ファイルを開いてBase64エンコードする
    with open(file_path, 'rb') as mp4_file:
        mp4_data = mp4_file.read()
        mp4_data_base64 = base64.b64encode(mp4_data).decode('utf-8')
    data1 = {
        'ADD_MOVIE': {
            'FILE_NAME': "Text Sweep.mp4",  # 
            'MOVIE_DATA': mp4_data_base64,  # Base64エンコードされたデータ
        }
    }

    #POST
    post2microAPP(ipaddr, data1)


        # スポットとして使う
    data2 = {'SPOT_PARAMS': [
        {'ID': 1, 
         'STATE': True,
         'POSX': 500,
         'POSY': 500,
         'SIZ': 1000,
         'BRI': 100,
         'COR': 0,
         'BLR': 0,
         'MOV': "Text Sweep.mp4",
         'MOV_FIT_TYPE': 0, # 0: Canvas-Fit, 1: Spot-Fit
         'IMG': 'default'
        }]}
    
    # スポット描画
    post2microAPP(ipaddr, data2)

    return


def send_video(ipaddr,file_path):

    with open(file_path, 'rb') as mp4_file:
        mp4_data = mp4_file.read()
        mp4_data_base64 = base64.b64encode(mp4_data).decode('utf-8')
    data1 = {
        'ADD_MOVIE': {
            'FILE_NAME': "Text Sweep.mp4",  # 
            'MOVIE_DATA': mp4_data_base64,  # Base64エンコードされたデータ
        }
    }

    #POST
    post2microAPP(ipaddr, data1)


        # スポットとして使う
    data2 = {'SPOT_PARAMS': [
        {'ID': 1, 
         'STATE': True,
         'POSX': 500,
         'POSY': 500,
         'SIZ': 1000,
         'BRI': 100,
         'COR': 0,
         'BLR': 0,
         'MOV': "Text Sweep.mp4",
         'MOV_FIT_TYPE': 0, # 0: Canvas-Fit, 1: Spot-Fit
         'IMG': 'default'
        }]}
    
    # スポット描画
    post2microAPP(ipaddr, data2)


def send_image_reveal(ipaddr,image_path,speed=1,brightness=1,window_width=10):
    # params : 
    # speed 
    # Width of window
    # Brightness
    file_path = text_reveal(image_path,"/home/jazz/Harish_ws/Demo/microled/agents/temp_storage",speed= speed,brightness=brightness,window_width = window_width)
    print("file path is in " , file_path)
    # ファイルを開いてBase64エンコードする
    with open(file_path, 'rb') as mp4_file:
        mp4_data = mp4_file.read()
        mp4_data_base64 = base64.b64encode(mp4_data).decode('utf-8')
    data1 = {
        'ADD_MOVIE': {
            'FILE_NAME': "Text Sweep.mp4",  # 
            'MOVIE_DATA': mp4_data_base64,  # Base64エンコードされたデータ
        }
    }

    #POST
    post2microAPP(ipaddr, data1)


        # スポットとして使う
    data2 = {'SPOT_PARAMS': [
        {'ID': 1, 
         'STATE': True,
         'POSX': 500,
         'POSY': 500,
         'SIZ': 1000,
         'BRI': 100,
         'COR': 0,
         'BLR': 0,
         'MOV': "Text Sweep.mp4",
         'MOV_FIT_TYPE': 0, # 0: Canvas-Fit, 1: Spot-Fit
         'IMG': 'default'
        }]}
    
    # スポット描画
    post2microAPP(ipaddr, data2)

    return

def clear_canvas(ipaddr):

    """

    clears the simulator canvas with a black image

    """
    img_path = "/home/jazz/Harish_ws/Demo/microled/agents/temp_storage/clear.png"
    image = Image.open(img_path)
    image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    data = {
        "BASE64_IMG": image_base64
    }

    post2microAPP(ipaddr, data)



def reshape_image(input_image_path, new_width=256, new_height=64):
    """
    Resizes an image to (new_width x new_height) and saves it 
    in the same directory with a modified filename.
    
    :param input_image_path: The path to the original image file.
    :param new_width: The width of the resized image (default=256).
    :param new_height: The height of the resized image (default=64).
    """
    
    # Extract directory, filename, and extension
    dir_name, file_name = os.path.split(input_image_path)
    name, ext = os.path.splitext(file_name)
    
    # Create a new output filename, e.g. original -> original_resized
    output_image_path = os.path.join(
        dir_name,
        f"{name}_resized{ext}"
    )
    
    # Open the image
    with Image.open(input_image_path) as img:
        # Resize the image
        # Note: This will not preserve aspect ratio.
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Save to new path
        resized_img.save(output_image_path)
        print(f"Resized image saved to: {output_image_path}")






if __name__ == "__main__":
    clear_canvas(IP)
    image_Post_sample(IP) #画像をポストするサンプル
    #draw_Spot_sample(IP) #スポットを表示するサンプル
    #send_Animation_localdata(IP) #アニメーションデータファイルを読み込んでポストするサンプル
    #send_Animation_data(IP) #アニメーションを生成しポストするサンプル
    #embed_image_into_spot_sample(IP)
    #embed_mp4file_into_spot_sample(IP)
    # draw_Text_sample(IP)
    # embed_mp4file_into_spot_sample(IP)
    # send_image_sweep(IP,"Test.png")
    # send_image_sweep(IP,"/home/lv_ia/harish_ws/microled/microAPP_Simulator_ver1.1.8/api/data/PATH_resized.png",)
    # image_path_sample(IP,'data/SMILE.jpg')
    # reshape_image("/home/lv_ia/harish_ws/Hi-SAM/testing/images/img643.jpg")
