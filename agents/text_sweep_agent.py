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

import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.append(os.path.join(root_dir, "Hi-sam"))
from text_sweep import text_sweep , text_reveal

import json
import re
from dotenv import load_dotenv





load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
hf_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")



prompt = PromptTemplate(input_variables=["history", "input"],
                            template="""

    You are an AI assistant that highlights text and traces over them based on natural language commands.


    The following conversation provides context:
    {history}

    Now, given the latest user command:
    {input}


    Your task is to convert these commands into values for speed of tracing, width of the tracing window, and the brightness. Follow these structured steps for every input:

    1. **Observation**: 
       - Clearly identify the user’s explicit instructions.
       - Note any implicit details like default values for unspecified parameters.
       - Always state the **Width of the tracing window and the Speed of tracing**. If it is not mention you must use the default values.
       - If the user previously changed the parameters, you should prioritize keeping these parameters untill the user explicitly changes them. 

    2. **Thought**: 
       - Describe your thought process for how you will interpret the user's request. 
       - If details are missing, explain what defaults or assumptions you will use.

    3. **Action**: 
       - Provide a list of dictionaries where each entry represents a variable to do the text sweep. 
       - Each dictionary must include the following keys and their respective values:
         - "speed of tracing": Decimal (1 to 10)
         - "width of the tracing window": integer (0 to 100)
         - "brightness": integer (0 to 10)
       

    Here are the default values for each of these variables : 

         - "speed of tracing": 1
         - "width of the tracing window": 20
         - "brightness": 8



    Example output format:

    - **Example 1**:  
    query: "Trace over this image's words"
    - Observation: The user wants to highlight the text in the image and traces over them, with no specific brightness, window width , or speed of tracing mentioned. I will assume default values for unspecified parameters.
    - Thought: I will Trace over the words in this image, use an average window width , and apply default brightness and speed of tracing.
    - Action: [
        {{"speed of tracing": 1, "width of the tracing window": 20, "brightness": 8}}

    ]

    - **Example 2**:  
    query: "Trace over this image's words slowly."
    - Observation: The user wants to highlight the text in the image and traces over them in a slow manner.However there is no specific brightness or window width mentioned. I will assume default values for unspecified parameters.
    - Thought: I will position adjust the speed of tracing to a low value, below the default value. I will apply default brightness and window width.
    - Action: [
         {{"speed of tracing": 0.5 , "width of the tracing window": 20, "brightness": 8}}
        
    ]

    - **Example 3**:  
    query: "Trace over this image's words with very low brightness."
    - Observation: The user wants to highlight the text in the image and traces over them, with low brightness. However there is no specific speed of tracing or window width mentioned. I will assume the default values for these unspecified parameters.
    - Thought: I will Trace over the words in this image, adjust the brightness to a low value, significantly lower than the default value as asked. I will apply default values for the speed of tracing and the window_width.
    - Action: [
        {{"speed of tracing": 1 , "width of the tracing window": 20, "brightness": 3}}

    ]

    - **Example 4**:  
    query: "Trace over this image's words with a large width."
    - Observation: The user wants to highlight the text in the image and traces over them, with large a window width. However there is no speicific speed of tracing or brightness mentioned. I will assume the default values for these unspecified parameters.
    - Thought: I will Trace over the words in this image, adjust the window width to a larger than default value as asked. I will apply default values for the speed of tracing and the brightness.
    - Action: [
        {{"speed of tracing": 1 , "width of the tracing window": 50 , "brightness": 8}}
    ]

    - **Example 5**:  
    query: "Trace over this image's words with a large width in a very fast manner."
    - Observation: The user wants to highlight the text in the image and traces over them, with large a window width. The user also wants the speed of tracing to be fast. Since no other parameters are specified, I will use default brightness.
    - Thought: I will Trace over the words in this image, adjust the window width to a larger than default value as asked. I will also adjust the speed of tracing to be larger than the default value. I will apply default values for brightness.
    - Action: [
            {{"speed of tracing": 5 , "width of the tracing window": 50 , "brightness": 8}}


    ]

    - **Example 6**:  
    query: "Trace over this image's words with a small width but make it very bright."
    - Observation: The user wants to highlight the text in the image and traces over them, with smaller window width. The user also wants it to be brighter than the default value. Since no other parameters are specified, I will use default speed of tracing.
    - Thought: I will Trace over the words in this image, adjust the window width to be smaller than the default value as asked. I will also adjust the brightness to be larger than the default value. I will apply default values for speed of tracing.
    - Action: [
            {{"speed of tracing": 1 , "width of the tracing window": 5 , "brightness": 10}}


    ]

    - **Example 7**:  
    
    query: Highlight all text in the image using a medium speed and moderate brightness, while maintaining a standard window width for better clarity. 
    - Observation: The user wants to highlight all the text in the image at a moderate pace, with moderate brightness, and a standard window width. There are no specific values given for speed, window width, and brightness. I need to deduce the appropriate settings from the context provided.
    - Thought: Since 'medium' speed and 'moderate brightness' are qualitative terms, we can infer reasonable quantitative values for these parameters within the allowed range of 0 to 10. Similarly, 'standard window width' implies a typical setting rather than an extreme one. Therefore, I would select mid-range values that reflect a balanced approach for both speed and brightness while ensuring a wide enough window width for clarity without being too wide.
    - Action: [
    {{"speed of tracing": 5, "width of the tracing window": 15, "brightness": 5}}
    ]

    
    - **Example 8**:  
    
    query: can u help me trace my document with high precision and at slower pace?
    - Observation: The user wants to trace over his document with high precision and at a slower pace. This implies two actions: increasing precision (default value assumed) and decreasing speed.
    - Thought: Higher precision can be interpreted as narrower window width. Decrease in speed means using a slower rate of tracing, implying a lower speed of tracing value.
    - Action: [
    {{"speed of tracing": 0.5 , "width of the tracing window": 5, "brightness": 8}}
    ]


    It is crucial to deeply understand the user's intent by analyzing both their current query and the context provided by previous interactions.If the user didn't ask you to explicitly change certain 
    parameters from their previous query, you can keep it constant. Here are some examples explaining this: 


    - **Example 9**:  

    - **First Query**: "Trace over this image's words with very low brightness."  
        - **Thought**: The user wants a significantly reduced brightness while other parameters remain at their default values.  
        - **Action**: [  
            {{"speed of tracing": 1, "width of the tracing window": 20, "brightness": 3}}  
        ]  
    - **Second Query**: "Now trace them quickly."  
        - **Thought**: The user now wants a faster tracing speed. Since no change is requested for brightness or window width, we retain the previous low brightness and default width.  
        - **Action**: [  
            {{"speed of tracing": 5, "width of the tracing window": 20, "brightness": 3}}  
        ]

    - **Example 10**:  

    - **First Query**: "Trace over this image's words with a small width."  
        - **Thought**: The user specifies a smaller tracing window while leaving the speed and brightness at their default values.  
        - **Action**: [  
            {{"speed of tracing": 1, "width of the tracing window": 5, "brightness": 8}}  
        ]  
    - **Second Query**: "Now trace them with a slower pace."  
        - **Thought**: The user is asking to slow down the tracing speed. We keep the previously set small window width and default brightness since they weren’t altered.  
        - **Action**: [  
            {{"speed of tracing": 0.5, "width of the tracing window": 5, "brightness": 8}}  
        ]


    Now return your answer accordingly : 

    query: {input}
    Observation: [PLACEHOLDER]
    Thought: [PLACEHOLDER]
    Action: [PLACEHOLDER]
    """)




prompt2 = PromptTemplate(input_variables=["history", "input"],
                            template="""

    You are an AI assistant.
    {input}

    This is the past conversation : 
    {history}

    """)


llm = ChatOpenAI(
    model_name="gpt-4-turbo",
    temperature=0,
    openai_api_key=openai_key,
    max_tokens=512
)



memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# Create a ConversationChain that uses your existing LLM and the memory
conversation = ConversationChain(
    llm=llm,
    prompt=prompt,
    memory=memory
)




#### ===================================================================================== API Part ===================================================================================== ####


IP = 'localhost'
import sys
import os




def get_paramDict(response):
    default_action = [{"speed of tracing": 1, "width of the tracing window": 10, "brightness": 8}]
    cleaned_response = response.replace("- **Action**:", "Action:")
    action_pattern = r"Action:\s*(\[[\s\S]*\])"
    match = re.search(action_pattern, cleaned_response)
    if match:
        action_text = match.group(1)
        try:
            action_vars = json.loads(action_text)
            print("Extracted action variables:", action_vars)
            return action_vars
        except json.JSONDecodeError as e:
            print("Error decoding action variables:", e)
    else:
        print("No action part found in the response")
    return default_action







def process_action(action_vars,image_dir,IP_addr,output_dir,mode = "Sweep"):
    # Process each action dictionary in the list
    for action in action_vars:
        speed = action.get("speed of tracing")
        width = action.get("width of the tracing window")
        brightness = action.get("brightness")
        # Now use these variables as needed; here we just print them

    if mode == "Sweep" :
        file_path = text_sweep(image_dir,output_dir,speed= speed,brightness=brightness,window_width = width)
    elif mode == "Reveal" :
        file_path = text_reveal(image_dir,output_dir,speed= speed,brightness=brightness,window_width = width)






def run_agent_text_sweep(prompt,image_dir,IP_addr,output_dir):
    
    response = conversation.predict(input=prompt)
    # print('RESPONSE ==>', response)
    param_dict = get_paramDict(response)
    # print(param_dict)
    process_action(param_dict,image_dir,IP_addr,output_dir)


def run_agent_text_reveal(prompt,image_dir,IP_addr,output_dir):
    
    response = conversation.predict(input=prompt)
    # print('RESPONSE ==>', response)
    param_dict = get_paramDict(response)
    # print(param_dict)
    process_action(param_dict,image_dir,IP_addr,output_dir,mode = "Reveal")






# while True:
#     input_message = input('User: ')
#     response = conversation.predict(input=input_message)

#     print('RESPONSE ==>', response)
#     param_dict = get_paramDict(response)
#     print(param_dict)
#     process_action(param_dict)
    
    
