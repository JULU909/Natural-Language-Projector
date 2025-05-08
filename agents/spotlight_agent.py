import ast
import re
import json
from uu import Error

import requests

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai.chat_models import ChatOpenAI
import os 
from dotenv import load_dotenv
load_dotenv()


openai_key = os.getenv("OPENAI_API_KEY")
hf_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def SpotLights(llm):
    prompt = PromptTemplate(input_variables=["query"],
                            template="""

    You are an AI assistant that draws circle-like spotlights based on natural language commands. 
    Your task is to convert these commands into values for position (X, Y), size, corner rounding, brightness, blur, 
    and the **number of spotlights** to generate. Follow these structured steps for every input:

    1. **Observation**: 
       - Clearly identify the user’s explicit instructions.
       - Note any implicit details like default values for unspecified parameters.
       - Always state the **number of spotlights** to generate, defaulting to 1 if not mentioned.

    2. **Thought**: 
       - Describe your thought process for how you will interpret the user's request. 
       - If details are missing, explain what defaults or assumptions you will use.

    3. **Action**: 
       - Provide a list of dictionaries where each entry represents a spotlight. 
       - Each dictionary must include the following keys and their respective values:
         - "size": integer (2 to 20)
         - "posX": integer (0 to 1000)
         - "posY": integer (0 to 1000)
         - "corner_rounding": integer (0 to 100)
         - "brightness": integer (10 to 100)
         - "blur": integer (10 to 100)
       - Additionally, include the key "num_spotlights" which indicates how many spotlights to generate.
       - Ensure the **num_spotlights** matches the number of entries in the list.

    Example output format:

    - **Example 1**:  
    query: "Draw a small circle in the center."
    - Observation: The user wants a small circle at the center, with no specific brightness, blur, or number of spotlights. I will assume one spotlight and default values for unspecified parameters.
    - Thought: I will place the circle at (500, 500), use a small size, and apply default brightness and blur.
    - Action: [
        {{"size": 4, "posX": 500, "posY": 500, "corner_rounding": 100, "brightness": 60, "blur": 50}},
        "num_spotlights": 1
    ]

    - **Example 2**:  
    query: "Create two large, bright spotlights near the top-left corner."
    - Observation: The user wants two bright, large spotlights near the top-left. Since two spotlights are requested, I will generate two.
    - Thought: I will position the spotlights near the top-left, adjust brightness to a high value, and use a large size.
    - Action: [
        {{"size": 18, "posX": 100, "posY": 100, "corner_rounding": 100, "brightness": 100, "blur": 30}},
        {{"size": 18, "posX": 150, "posY": 120, "corner_rounding": 100, "brightness": 100, "blur": 30}},
        "num_spotlights": 2
    ]

    - **Example 3**:  
    query: "Draw five random circles with different sizes and brightness values."
    - Observation: The user wants five circles with varying sizes and brightness. No positions were specified, so I will place them randomly across the canvas.
    - Thought: I will generate random sizes and brightness values for the five spotlights, with random positions.
    - Action: [
        {{"size": 6, "posX": 250, "posY": 400, "corner_rounding": 80, "brightness": 70, "blur": 40}},
        {{"size": 10, "posX": 500, "posY": 600, "corner_rounding": 60, "brightness": 90, "blur": 30}},
        {{"size": 14, "posX": 750, "posY": 200, "corner_rounding": 50, "brightness": 80, "blur": 20}},
        {{"size": 8, "posX": 300, "posY": 300, "corner_rounding": 90, "brightness": 60, "blur": 50}},
        {{"size": 12, "posX": 900, "posY": 700, "corner_rounding": 70, "brightness": 100, "blur": 60}},
        "num_spotlights": 5
    ]

    - **Example 4**:  
    query: "I want to draw two circles, one on the left and one on the right, with soft edges."
    - Observation: The user wants two circles with soft edges, one on the left and one on the right. I will set the blur to a high value to achieve soft edges.
    - Thought: I will position the circles on the far left and right sides and use higher blur values for softer edges.
    - Action: [
        {{"size": 12, "posX": 100, "posY": 500, "corner_rounding": 80, "brightness": 80, "blur": 80}},
        {{"size": 12, "posX": 900, "posY": 500, "corner_rounding": 80, "brightness": 80, "blur": 80}},
        "num_spotlights": 2
    ]

    - **Example 5**:  
    query: "Draw 3 small circles at random positions, each with different corner rounding."
    - Observation: The user wants three small circles with varying corner rounding values at random positions. Since no other parameters are specified, I will use default brightness and blur values.
    - Thought: I will randomly position the circles and vary the corner rounding as requested.
    - Action: [
        {{"size": 4, "posX": 300, "posY": 400, "corner_rounding": 30, "brightness": 50, "blur": 40}},
        {{"size": 4, "posX": 600, "posY": 700, "corner_rounding": 60, "brightness": 50, "blur": 40}},
        {{"size": 4, "posX": 800, "posY": 300, "corner_rounding": 90, "brightness": 50, "blur": 40}},
        "num_spotlights": 3
    ]

    query: {query}
    Observation: [PLACEHOLDER]
    Thought: [PLACEHOLDER]
    Action: [PLACEHOLDER]
    """)
    return LLMChain(llm=llm, prompt=prompt)



IP = 'localhost'
def post2microAPP(ipaddr, jsondata):
    url = 'http://' + ipaddr + ':8080'
    response = requests.post(url, json=jsondata)
    print(response.text)


key_mapping = {
    'size': 'SIZ',
    'posX': 'POSX',
    'posY': 'POSY',
    'corner_rounding': 'COR',
    'brightness': 'BRI',
    'blur': 'BLR'
}


def draw_Spot_samples(id, spotlight):

    spotdata = {}
    spotdata["ID"] = id
    spotdata["STATE"] = True
    spotdata["SIZ"] = spotlight.get('SIZ', 8)
    spotdata["POSX"] =  spotlight.get('POSX', 90)
    spotdata["POSY"] = spotlight.get('POSY', 500)
    spotdata["COR"] = spotlight.get('COR', 50)
    spotdata["BRI"] = spotlight.get('BRI', 50)
    spotdata["BLR"] = spotlight.get('BLR', 50)
    return spotdata


def send_Animation_localdata(ipaddr):

    jsonfile = '/home/raja/Documents/source_code/AIAgents/MicroLED/API/data/0_splash_newAPI.json'

    with open(jsonfile) as f:
        jsondata = json.load(f)
        data = {
            'SPOT_ANIMATION': jsondata
        }
        post2microAPP(ipaddr, data)

def draw_spotLights(response):
    response = response.replace('{', '[').replace('}', ']')
    print(response)
    print('-----------------------------')

    action_matches = re.search(r"Action:\s*(\[[\s\S]*?\])", response)
    print( action_matches)

    data = {}
    spot_params = []

    if action_matches:
        action_list = []
        for match in action_matches:
            properties = match[0]
            num_spotlights = int(match[1])
            properties_list = properties.split(',')
            action_dict = {}
            for prop in properties_list:
                key, value = prop.split(":")
                action_dict[key.strip().replace('"', '')] = int(value.strip())
            action_dict["num_spotlights"] = num_spotlights
            action_dict = {key_mapping.get(k, k): v for k, v in action_dict.items()}
            action_list.append(action_dict)
        print(action_list)

        for i, action in enumerate(action_list, start=1):
            spot_params.append(draw_Spot_samples(i, action))

        data["SPOT_PARAMS"] = spot_params
        post2microAPP(IP, data)




llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    max_tokens=512
)

spotlightAgent = SpotLights(llm)


def get_paramDict(response):
    match = re.search(r'Action:\s*(\[.*?\])', response, re.DOTALL)
    if match:
        action_data = match.group(1)
        cleaned_string = re.sub(r',?\s*"num_spotlights":\s*\d+.*$', '', action_data, flags=re.DOTALL)

        if cleaned_string and not (cleaned_string.startswith('[') and cleaned_string.endswith(']')):
            cleaned_string = f'[{cleaned_string}]'

        cleaned_string = cleaned_string.replace('[[', '[').replace(']]', ']')

        params = '{"params":' + cleaned_string + "}"
        params_dict = json.loads(params)
        return params_dict
    else:
        print('Something went wrong')


def replace_keys(original_dict, key_mapping):
    new_dict = {'SPOT_PARAMS': []}
    for index, item in enumerate(original_dict['params']):
        new_item = {}
        for old_key, new_key in key_mapping.items():
            if old_key in item:
                new_item[new_key] = item[old_key]

        new_item['ID'] = index + 1
        new_item['STATE'] = True
        new_dict['SPOT_PARAMS'].append(new_item)

    return new_dict


def run_spolight_agent(prompt,ip_addr):
    response = spotlightAgent.invoke(prompt)['text']
    print('RESPONSE ==>', response)
    param_dict = get_paramDict(response)
    data = replace_keys(param_dict, key_mapping)
    return data
    # post2microAPP(ip_addr, data)


# while True:
#     input_message = input('User: ')
#     response = spotlightAgent.invoke(input_message)['text']

#     print('RESPONSE ==>', response)
#     param_dict = get_paramDict(response)
#     data = replace_keys(param_dict, key_mapping)
#     post2microAPP(IP, data)

# input_message = 'draw a cirlce at the right side'
# response = spotlightAgent.invoke(input_message)['text']
# param_dict = get_paramDict(response)
# data = replace_keys(param_dict, key_mapping)
# post2microAPP(IP, data)

