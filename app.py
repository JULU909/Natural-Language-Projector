
import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint
from huggingface_hub import login
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import shutil
import json
import re
import numpy as np
from dotenv import load_dotenv
import time
from agents.image_generation_agent import run_agent_image_generation 
from agents.text_sweep_agent import run_agent_text_sweep ,run_agent_text_reveal
from agents.spotlight_agent import run_spolight_agent 
from agents.perspective_transform import camera_transform , calibration , capture_input
from agents.task_planner_agent import run_agent_task_planner , prompt
from api.API_sample import send_image , image_path_sample , send_video , post2microAPP , post_all_off


load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
hf_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")


IP = '192.168.2.138'

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
image_database = os.path.join(BASE_DIR, "agents", "temp_storage")
send = []




def image_generation(prompt,image_dir):
    save_path = os.path.join(image_dir ,"generated_image.png")
    run_agent_image_generation(IP_addr=IP,prompt=prompt,save_dir=save_path)
    send.append(["image",IP,save_path])
    # send_image(IP,save_path)


def text_sweep(prompt,image_dir):
    image_dir = os.path.join(image_dir ,"camera_input.png")
    if os.path.exists(image_dir) == False:
        image_dir = os.path.join(image_dir ,"calibration_image.png")
    run_agent_text_sweep(prompt=prompt,image_dir=image_dir,IP_addr=IP,output_dir=image_database)
    vid_path = os.path.join(image_database ,"video.mp4")
    send.append(["video",IP,vid_path])
    # send_video(IP,vid_path)
    print(f"Text Sweep for image has been finished!")

def letter_trace(prompt,image_dir):
    print(f"Text Sweep for image has been finished! in {image_dir}")

def text_reveal(prompt,image_dir = image_database) : 
    image_dir = os.path.join(image_dir ,"camera_input.png")
    if os.path.exists(image_dir) == False:
        image_dir = os.path.join(image_dir ,"calibration_image.png")
    run_agent_text_reveal(prompt=prompt,image_dir=image_dir,IP_addr=IP,output_dir = image_database)
    vid_path = os.path.join(image_database ,"video.mp4")
    # print(["video",IP,vid_path])
    send.append(["video",IP,vid_path])
    # send_video(IP,vid_path)
    print(f"Text reveal is done.")


def spotlights(prompt,image_dir):
    print(f"runnint spotlights done")
    # data = run_spolight_agent(prompt,ip_addr= IP)
    # send.append(["spotlight",IP,data])
    # post2microAPP(ip_addr, data)

def camera_input(prompt, image_dir = image_database):
    points_file = os.path.join(image_dir, "calibration_points.json")
    
    if not os.path.exists(points_file):
        print("Error: Calibration points file not found.")
        return
    
    try:
        with open(points_file, "r") as f:
            points = json.load(f)
        
        save_path = os.path.join(image_dir, "camera_input.png")
        capture_input(src_points=points, save_path=save_path)
        print("Camera input applied")
    
    except Exception as e:
        print(f"Error: {e}")
        return



def calibrate(image_dir = image_database):
    send_image(IP,"agents/data/grid.bmp")
    save_path = os.path.join(image_dir,"calibration_image.png")
    global points
    points = calibration(save_path=save_path)

    if isinstance(points, np.ndarray):
        points = points.tolist()  # Convert NumPy array to a list

    points_file = os.path.join(image_database, "calibration_points.json")
    with open(points_file, "w") as f:
        json.dump(points, f)
    print(f"Calibration points saved to {points_file}")
    print(f"Calibration is done for user")
    print(points)
    camera_input_path = os.path.join(image_dir, "camera_input.png")
    # Copy the file
    shutil.copy(save_path, camera_input_path)
    clear_canvas()
    print(f"Calibration image copied to {camera_input_path}")

    return

def clear_canvas():
    post_all_off(IP)
    print("canvas is being cleared.")


def send_all():
    global send 

    for call in send :
        if call[0] == "image":
            send_image(call[1],call[2])
        elif call[0] == "video": # video api call sometimes can cause a buffer or something so redo 
        
            send_video(call[1],call[2])
            send_video(call[1],call[2])

        elif call[0] == "spotlight":
            print(call[2],call[1])
            post2microAPP(call[1], call[2])


    send.clear()
    return




def get_paramDict(response):
    """
    Extracts the 'Action' list from the response using regex and converts it to a structured dictionary.

    Args:
        response (str): The full response containing 'Observation', 'Thought', and 'Action' sections.

    Returns:
        list: A list of dictionaries extracted from the 'Action' section.
    """
    try:
        # First, try to parse the response as JSON directly
        try:
            response_json = json.loads(response)
            if "Action" in response_json and isinstance(response_json["Action"], list):
                return response_json["Action"]  # Directly return the Action list
        except json.JSONDecodeError:
            pass  # If it's not valid JSON, continue with regex extraction

        # Regex to match **Action**, ** Action **, ACTION, or Action (case-insensitive)
        action_match = re.search(r"(?i)\b\**\s*Action\s*\**\s*:\s*(\[[\s\S]*?\])", response)

        if not action_match:
            print(f"❌ Error: 'Action' section not found in the response.\nResponse snippet: {response[:200]}...")
            return []

        action_json = action_match.group(1).strip()  # Extract JSON-like list
        
        try:
            return json.loads(action_json)  # Convert to Python list of dictionaries
        except json.JSONDecodeError as e:
            print(f"❌ Error: Invalid JSON format in extracted Action section.\nDetails: {e}\nExtracted text: {action_json[:200]}...")
            return []

    except Exception as e:
        print(f"❌ Unexpected error occurred: {e}")
        return []


def process_action(action_vars, default_image_dir=image_database):
    """
    Processes actions based on function names from action_vars.
    Tracks and updates image locations dynamically for subsequent actions.
    """

    current_image_dir = default_image_dir  # Default image storage path

    for action in action_vars:
        function_name = action.get("Function name")
        prompt = action.get("Prompt")
        image_location = default_image_dir 
        if function_name == "image_generation":
            # Generate new image and update the current image location
            current_image_dir = image_generation(prompt, image_dir=image_location)

        elif function_name == "text_sweep":
            # Ensure image exists before applying text_sweep
            if os.path.exists(image_location):
                text_sweep(prompt, image_location)
            else:
                print(f"Error: No image found at {image_location} for text_sweep!")

        elif function_name == "letter_trace":
            # Ensure image exists before applying letter_trace
            if os.path.exists(image_location):
                letter_trace(prompt, image_location)
            else:
                print(f"Error: No image found at {image_location} for letter_trace!")

        elif function_name == "spotlights":
            # Ensure image exists before applying spotlight effect
            if os.path.exists(image_location):
                spotlights(prompt, image_location)
            else:
                print(f"Error: No image found at {image_location} for spotlights!")

        elif function_name == "clear_canvas":
            clear_canvas()

        elif function_name == "camera_input":
            camera_input(prompt,image_location)

           
        elif function_name == "text_reveal":
            text_reveal(prompt,image_location)

        else:
            print(f"Unknown function: {function_name}")

    return current_image_dir  # Return the last updated image location




llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=openai_key,
    max_tokens=512,
    seed = 42, 
)



memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# Create a ConversationChain that uses your existing LLM and the memory
conversation = ConversationChain(
    llm=llm,
    prompt=prompt,
    memory=memory
)




if __name__ == "__main__" : 
    calibration_file = os.path.join(image_database,"calibration_image.png") # Change this to the actual file you use

    if os.path.exists(calibration_file):
        user_input = input("⚠️ Calibration is already done. Do you want to recalibrate? (Yes/No): ").strip().lower()
        if user_input in ["yes", "y"]:
            calibrate()
            send_all()
        else:
            print("✅ Skipping calibration.")
    else:
        print("⚠️ No previous calibration found. Running calibration now...")
        calibrate()
        send_all()
    while True:
        input_message = input(
            '''
            This is a MicroLED Demo.
            It makes use of multiagent LLM to produce desired lighting effects.
            Current available effects are:
                1, Image Generation
                2, Text Sweep
                3, Text Reveal
            User:
            '''
        )
        t1 = time.time()
        response = run_agent_task_planner(llm,conversation,memory,input_message)
        print('RESPONSE ==>', response)
        param_dict = get_paramDict(response)
        print(param_dict)
        process_action(param_dict)
        t2 = time.time()
        send_all()
        print(f"time taken : {t2-t1}")