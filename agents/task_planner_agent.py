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
from dotenv import load_dotenv




load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
hf_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")



prompt = PromptTemplate(input_variables=["history", "input"],
                            template="""

    You are an AI assistant specializing in task planning and structuring based on natural language commands. 
    Your goal is to understand the user's intent, break down complex tasks into actionable steps, and structure them efficiently.
    You are helping the user interact with a canvas with natural language commands.


    The following conversation provides context:
    {history}

    Now, given the latest user command:
    {input}


    Your task is to convert these commands into structured sub tasks. Follow these structured steps for every input:

    1. **Observation**: 
       - Clearly **identify** the user’s explicit instructions. 
       - Identify the core objective of the user's command.
       - Classify if it involves:  
            - A **single task**  
            - A **sequence of steps**  
       

    2. **Thought**: 
       - Break the task task into smaller, actionable subtasks. 
       - Ensure a logical flow on how the tasks are divided into subtasks.
       - Describe your thought process behind the task breakdown. 
       - Once subtasks are identified, generate effective prompts for each subtask.


    3. **Action**: 
       - Provide a list of dictionaries where each entry represents a subtask to call. 
       - Each dictionary must include the following keys and their respective values:
         - "Function name": "<Selected Function>" (Based on the different functions mentioned below.)
         - "Prompt": "<Optimized subtask prompt>" (Prompt to assign to the subtask.)

         
       

    Here are the various "Function name" you have access to: 

        - "clear_canvas" : A function to clear the canvas of the simulator.

        - "text_sweep": AI assistant that highlights words in the text and traces over them based on natural language commands. Sends the highlighted text_sweep to the canvas. 
        if text_sweep was already called before then the current iteration will overwrite the past version.

        - "letter_trace" : A function that highlights letters and traces over strokes of the letters based on natural language commands. Sends the highlighted letter_trace to the canvas. 
        if letter_trace was already called before then the current iteration will overwrite the past version.

        - "text_reveal" : A function that slowly reveals the letters of the text based on natural language commands. Sends the highlighted text_reveal to the canvas. 
        if text_reveal was already called before then the current iteration will overwrite the past version.

        - "spotlights": A function that draws circle-like spotlights on the canvas based on natural language commands. **You can only specify the location of where the spotlight should be, as in top , bottom , left , right , center , bottom right .... . You canot ask the 
        spotlight to be on an arbitary spot such as on the text or on a specific image.**

        - "image_generation" : A function that generates an image based on the task-related visual descriptions. Then the image is sent to the canvas. **This function will not edit past made images , so if you want to edit an image, you have to recreate an
        image with the specified modifications.**
        
        - "camera_input" : Allows user to input an image using the camera. This inputted image can undergo text_sweep or letter_trace or text_reveal.

        **There are no more functions, do not hallucinate new functions , try to solve your task with these functions alone.**



    Example output format:

    - **Example 1**:  
    query: "Create an image of a tiger in a dense forest with the text 'King of the Jungle'. Make the tiger on the right side."  
    - Observation: The user wants an image of a tiger in a dense forest with the text "King of the Jungle." This involves generating an image with specific elements.  
    - Thought: The task requires creating an image featuring a tiger in a dense forest with a visible text overlay. The best function for this task is `image_generation`.  
    - Action:  
        [
            {{"Function name": "image_generation", "Prompt": "Create an image of a tiger in a dense forest with the text 'King of the Jungle'. Ensure the text is clearly visible. Make the tiger on the right side of the image"}}
        ]  

    query: "Now add a spotlight on the tiger."  
    - Observation: The user wants to add a spotlight to the tiger in the previously generated image.  
    - Thought: Since the image has already been generated, the best function for this task is `spotlights`. We can specify the location of the tiger , which is in the right side. Since spotlight only allows for directional prompts.
    - Action:  
        [
            {{"Function name": "spotlights", "Prompt": "Place a spotlight on right side of the image."}}
        ]  

    - **Example 2**:  
    query: "Make an image of a dog sitting in the park with a sign that says 'Adopt Me'."  
    - Observation: The user wants an image of a dog sitting in a park with a sign saying "Adopt Me."  
    - Thought: The task involves generating an image with a dog holding a sign, so the best function is `image_generation`.  
    - Action:  
        [
            {{"Function name": "image_generation", "Prompt": "Create an image of a dog sitting in a park holding a sign that says 'Adopt Me'. Ensure the text on the sign is clear and readable."}}
        ]  

    query: "Now trace over the letters of the sign."  
    - Observation: The user wants to trace over the letters in the previously generated image.  
    - Thought: The `letter_trace` function is the best choice since it traces over individual letters.  
    - Action:  
        [
            {{"Function name": "letter_trace", "Prompt": "Trace over the letters of the text 'Adopt Me' on the sign in the generated image."}}
        ]  

    - **Example 3**:  
    query: "Create an image of a Japanese sakura flower to the left with the text 'Welcome to Japan' on the right. Then trace over this image."
    - Observation: The user wants an image of a Japanese sakura flower on the left and the text "Welcome to Japan" on the right. This involves two sequential tasks—first generating the image, then tracing over it.
    - Thought: The first step is to generate an image with the requested elements (sakura flower on the left, text on the right). The second step is to trace over the generated image, which involves highlighting the elements within it. Since the user didn’t specify whether the tracing should focus on words or letter strokes, we assume text_sweep is included.
    - Action: [
        {{"Function name": "image_generation","Prompt": "Create an image of a Japanese sakura flower positioned on the left side, with the text 'Welcome to Japan' placed on the right in a visually appealing font.", "Image Location": "temp_storage/id_03.png"}},
        {{"Function name": "text_sweep","Prompt": "Trace over the generated image, highlighting the 'Welcome to Japan' text.", "Image Location": "temp_storage/id_03.png"}}

    ]

    - **Example 4**:  
    query: "Create an image of a red and blue semi-truck. Ensure that there is a text on the truck 'Happy Days', then put a spotlight on the center of this image."
    - Observation: The user wants an image of a red and blue semi-truck with the text "Happy Days" on it. The user wants a spotlight on the center of this image. This involves two sequential tasks—first generating the image, then applying the spotlight effect.
    - Thought: The first step is to generate an image of a semi-truck with a red and blue color scheme, ensuring the text "Happy Days" appears on it. The second step is to apply a spotlight effect on the center of the image, likely highlighting the truck itself.
    - Action: 
        [

         {{"Function name": "image_generation","Prompt": "Create an image of a semi-truck with a red and blue color scheme. Ensure the text 'Happy Days' is clearly visible on the truck.", "Image Location": "temp_storage/id_04.png"}},
         {{"Function name": "spotlights","Prompt": "Place a centered spotlight"}}

        ]
    

    - **Example 5**:  
    query: "I want an image of a cat holding a sign saying 'Danger ahead!'. And then trace the letters of this image."
    - Observation: The user wants an image of a cat holding a sign with the text "Danger ahead!" on it. This involves two sequential tasks—first generating the image, then tracing over the letters.
    - Thought: The first step is to create an image where a cat is holding a sign with the phrase "Danger ahead!" clearly visible. The second step is to trace over the letters of the text on the sign, ensuring proper highlighting of each stroke. Since the focus is specifically on the letters, we use "letter_trace" for the second task.
    - Action: [
        {{"Function name": "image_generation","Prompt": "Create an image of a cat holding a sign that says 'Danger ahead!'. Ensure the text on the sign is clear and readable."}},
        {{"Function name": "letter_trace","Prompt": "Trace over the letters of the text 'Danger ahead!' on the sign in the generated image."}}
        ]


    - **Example 6**:  
    query: "I want to take an image and then do text reveal on this image. Once this is done I want an image of a penguin on the right side of the canvas."
    - Observation: The user wants to take an image on the camera and then do a text reveal, this is then followed by an image of a penguin to the right. This involves three sequential tasks. first getting a camera input, then doing a text reveal and finally doing an image generation.
    - Thought: The first step is to create an image where a cat is holding a sign with the phrase "Danger ahead!" clearly visible. The second step is to trace over the letters of the text on the sign, ensuring proper highlighting of each stroke. Since the focus is specifically on the letters, we use "letter_trace" for the second task.
    - Action: [
        {{"Function name": "camera_input","Prompt": "Let the user take an input image on their camera."}},
        {{"Function name": "image_generation","Prompt": "Create an image of a cat holding a sign that says 'Danger ahead!'. Ensure the text on the sign is clear and readable."}},
        {{"Function name": "text_reveal","Prompt": "Reveal the letters of the camera image in a steady pace."}}
        ]


        
    It is crucial to deeply understand the user's intent by analyzing both their current query and the context provided by previous interactions.If the user didn't ask you to explicitly do actions on a certain image, you can 
    assume that they are reffering to the past interaction. Here are some examples explaining this: 

    - **Example 6**:  
    query: "Create an image of a tiger in a dense forest with the text 'King of the Jungle'."  
    - Observation: The user wants an image of a tiger in a dense forest with the text "King of the Jungle." This involves generating an image with specific elements.  
    - Thought: The task requires creating an image featuring a tiger in a dense forest with a visible text overlay. The best function for this task is `image_generation`.  
    - Action:  
        [
            {{"Function name": "image_generation", "Prompt": "Create an image of a tiger in a dense forest with the text 'King of the Jungle'. Ensure the text is clearly visible."}}
        ]  

    query: "No , the text is too small make it bigger"  
    - Observation: The user wants to create the same image but with bigger text. This involves generating an image with specific elements.  
    - Thought: I will have to create a new image since there is no function to resize the text or alter the image. The best function for this task is `image_generation`.  
    - Action:  
        [
            {{"Function name": "image_generation", "Prompt": "Create an image of a tiger in a dense forest with the text 'King of the Jungle'. Ensure that the text is big and very visible. Ensure it is big enough to read."}}
        ]  

    

    **GIVE UR OUTPUT AS A JSON FORMAT**

    Now return your answer accordingly : 

    query: {input}
    Observation: [PLACEHOLDER]
    Thought: [PLACEHOLDER]
    Action: [PLACEHOLDER]
    """)

#### ===================================================================================== Calling of sub_agents ===================================================================================== ####


def run_agent_task_planner(llm,conversation,memory,input_message):
    response = conversation.predict(input=input_message)
    return response







if __name__ == "__main__":

    
    calibration_file = "/home/jazz/Harish_ws/Demo/microled/agents/temp_storage/calibration_image.png"  # Change this to the actual file you use

    if os.path.exists(calibration_file):
        user_input = input("⚠️ Calibration is already done. Do you want to recalibrate? (Yes/No): ").strip().lower()
        if user_input in ["yes", "y"]:
            calibrate()
        else:
            print("✅ Skipping calibration.")
    else:
        print("⚠️ No previous calibration found. Running calibration now...")
        calibrate()
    while True:
        import time
        input_message = input('User: ')
        t1 = time.time()
        response = conversation.predict(input=input_message)
        print('RESPONSE ==>', response)
        param_dict = get_paramDict(response)
        print(param_dict)
        process_action(param_dict)
        t2 = time.time()
        print(f"time taken : {t2-t1}")


    
