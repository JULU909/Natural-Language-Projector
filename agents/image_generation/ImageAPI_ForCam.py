import base64
import os

import requests
import json
from PIL import Image
import io
import numpy as np
import cv2

import torch

from model import T5EncoderModel, FluxTransformer2DModel
from diffusers import FluxPipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai.chat_models import ChatOpenAI


class ImageAI:
    def __init__(self):
        text_encoder_2: T5EncoderModel = T5EncoderModel.from_pretrained(
            "HighCWu/FLUX.1-dev-4bit",
            subfolder="text_encoder_2",
            torch_dtype=torch.bfloat16,
            # hqq_4bit_compute_dtype=torch.float32,
        )

        transformer: FluxTransformer2DModel = FluxTransformer2DModel.from_pretrained(
            "HighCWu/FLUX.1-dev-4bit",
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )

        self.pipe: FluxPipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            text_encoder_2=text_encoder_2,
            transformer=transformer,
            torch_dtype=torch.bfloat16,
        )
        self.pipe.enable_model_cpu_offload()  # with cpu offload, it cost 8.5GB vram
        # self.pipe.remove_all_hooks()
        # self.pipe = self.pipe.to('cuda')  # without cpu offload, it cost 11GB vram


    def generate(self, prompt):
        image = self.pipe(
            prompt,
            height=256,
            width=1024,
            guidance_scale=5,
            output_type="pil",
            num_inference_steps=20,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(0)
        ).images[0]

        return image


GenAI = ImageAI()


def post2microAPP(ipaddr, jsondata):
    url = 'http://' + ipaddr + ':8080'
    response = requests.post(url, json=jsondata)
    print(response.text)

def image_Post_sample(image, ipaddr):
    print('Goes inside Posting')
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    data = {
        "BASE64_IMG": image_base64
    }
    print('About to pose =====>')
    post2microAPP(ipaddr, data)
    print('Posting done')
    return True

def highlight_text(ipaddr):
    pts1 = np.float32([[191, 186], [441, 206],
                       [178, 251], [445, 259]])

    pts2 = np.float32([[0, 0], [512, 0],
                       [0, 128], [512, 128]])

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    img = np.zeros((512, 128), dtype=np.uint8)
    cap = cv2.VideoCapture(4)
    h = 160
    w = 600
    while (1):
        ret, frame = cap.read()
        if not ret:
            break
        cv2.namedWindow('c', cv2.WINDOW_NORMAL)
        cv2.imshow('a', frame)
        # Press 'q' to exit the loop
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(frame, matrix, (512, 128))
        cv2.imshow('b', img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        gradient = cv2.dilate(gradient, kernel)
        gradient[gradient > 60] = 255
        gradient[gradient <= 50] = 0
        cv2.imshow('c', gradient) 
        image = Image.fromarray(gradient)

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
        if cv2.waitKey(1) == ord('q'):
            break

        return gradient


# prompt = "Generate an image of the text 'Hello World!' in black and white. The image should contain only pure black (hex #000000) and pure white (hex #FFFFFF) pixels, with no gray or intermediate shades. The text should be in a blocky, pixel-art style, centered on a white background, with each letter fully black and highly legible."
# image = GenAI.gcreate a big Japan Sakura flower create a sakura flower icon on the left of the imageenerate(prompt)
#
# image_Post_sample(image=image, ipaddr=ipaddr)


def AnimeAgent(llm):
    prompt = PromptTemplate(
        input_variables=["query"],
        template="""
        You are a highly creative assistant designed to transform user queries into specific, actionable text prompts for black-and-white image generation tools. Your task is to generate prompts for images that consist **exclusively** of pure black (hex #000000) and pure white (hex #FFFFFF) pixels, with no gray or intermediate tones.

        ### Guidelines:
        1. **Focus on Key Elements**: Analyze the query and identify critical components such as the subject, composition, and artistic style.
        2. **Strict Color Scheme**: Ensure the prompt specifies that the image must only contain black-and-white pixels with no gradients.
        3. **Artistic Style**: Suggest relevant styles such as pixel art, line art, high-contrast imagery, or bold typography.
        4. **Clarity and Specificity**: Make the output concise, clear, and highly detailed, ensuring the requirements are easy to follow.

        ### Example Inputs and Outputs:

        **Input**: "Create a black-and-white image of a cat playing a guitar."
        **Output**: "A highly detailed, black-and-white pixel art of a cat playing an acoustic guitar. The image should consist only of pure black and white pixels, with no gradients or gray tones. The cat is depicted in solid white on a black background, with the guitar in sharp black lines. The overall style is high contrast and reminiscent of vintage comics."

        **Input**: "Generate a black-and-white image of the word 'HELLO' in large block letters."
        **Output**: "A black-and-white image featuring the word 'HELLO' in bold, blocky capital letters. The letters are solid black on a pure white background, with no gradients or shades. The text should be centered and highly legible."

        **Input**: "Create a dynamic image of a futuristic city skyline."
        **Output**: "A sharp, high-contrast black-and-white image of a futuristic city skyline. The buildings should feature angular, geometric designs, depicted with solid black shapes against a pure white sky. Add intricate patterns to the building facades, using only black and white pixels. The overall style should evoke a retro-futuristic comic book aesthetic."

        **Input**: "Generate a pixel art poster of a forest in winter with animals."
        **Output**: "A pixel art image of a snowy winter forest with tall, leafless trees. Include small black silhouettes of forest animals such as deer and foxes on a white ground. Ensure the entire composition uses only black and white pixels for a bold and clear visual effect."

        Now, based on the user input, generate a prompt following the above guidelines:

        Query: {query}
        """
    )
    return LLMChain(llm=llm, prompt=prompt)

llm = ChatOpenAI(
    temperature=0,
    streaming=True,
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model_kwargs={"seed": 42}
)

spotlightAgent = AnimeAgent(llm)



ipaddr = '192.168.2.138'
fg = highlight_text(ipaddr)
fg = np.asarray(fg)[..., np.newaxis]
fg[fg > 50] = 255
fg[fg <= 50] = 0
while True:
    input_message = input('User: ' )

    if input_message == 'exit':
        break

    response = spotlightAgent.invoke(input_message)['text']
    print('LLM Agent generated prompt:', response)
    image = GenAI.generate(response)
    image.show()
    os.wait()

    img = np.asarray(image)
    img = cv2.resize(img, (512, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[..., np.newaxis]
    img = img.astype(float)
    fg = fg.astype(float)
    image = img + fg
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    image = image.astype(np.uint8)
    image = Image.fromarray(image)
    image.show()
    # image_Post_sample(image=image, ipaddr='localhost')
    while True:
        return_value = image_Post_sample(image=image, ipaddr=ipaddr)
        if return_value:
            break

# create a big Japan Sakura flower create a sakura flower icon on the left of the image
# create a big Japan sakura flower icon on the left of the image
# create sakura flower on the left side on the image. The flower should be in white and the background should be in black
# create christmas tree on the left side of the image. The christmas tree should be in white and the background should be in black