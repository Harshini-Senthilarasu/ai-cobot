import os
from dotenv import load_dotenv
import google.generativeai as genai
from enum import Enum
import json
import datetime
import re

load_dotenv(dotenv_path="/home/harshini/capstone/src/.env")
api_key = os.getenv("google_api_key")
data_files_path = os.getenv("convo_log_path")

class llmHandler:
    class TaskState(Enum):
        IDENTIFICATION = "IDENTIFICATION"
        MODIFICATION = "MODIFICATION"
        DETECTION = "DETECTION"
        EXECUTION = "EXECUTION"

    def __init__(self, logger):
        self.logger = logger
        # Setup Gemini 
        self.generation_config = {"temperature": 1, "top_p": 0.95, "top_k": 40, "max_output_tokens": 8192, "response_mime_type": "text/plain"}
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config=self.generation_config,
            system_instruction="""
            You are a Large Language Model connected to a cobot arm.
            Your goal is to process the user information in structured States. 
            These are the States:
            1. IDENTIFICATION - Extract the target item, goal position, and breakdown the task into clear steps. Ask the user for confirmation by saying: Reply 'yes' to proceed or 'no' to modify.
            2. MODIFICATION - If the user says no in IDENTIFICATION State, modify the task breakdown. Request for more details if required.
            3. DETECTION - If the user says yes in IDENTIIFCATION State, analyse the detected object list to find the closest possible object to the target item based on the label and the confidence score.
            4. EXECUTION - Once the target item is identified, change the State to this. 
            The starting State is IDENTIFICATION. Change the State if/when required.
            Rules:
            1. Ask: "Does this breakdown look correct? Reply 'yes' or 'no'."
            2. If the user says "no", modify accordingly.
            3. If the user says "yes", check for target item in detected objects list provided. Indicate the target item as such: Target: TARGET_ITEM_NAME.
            4. Report object detection results based on confidence scores.
            5. Always indicate what State as such: State: IDENTIFICATION
            """
        )
        self.convo_hist = []
        self.chat_session = self.model.start_chat(history=self.convo_hist)
        os.makedirs(data_files_path, exist_ok=True)
        self.current_state = self.TaskState.IDENTIFICATION
        self.execution_flag = False
        self.logger.info("LLM Handler started")

    def get_llm_response(self, user_prompt):
        self.add_to_convo_hist("user", user_prompt)
        response = self.chat_session.send_message(user_prompt)
        self.add_to_convo_hist("llm", response.text)
        llm_response = self.extract_state(response.text)
        self.logger.info(f"LLM response: {llm_response}")
        return llm_response
    
    def extract_target(self, llm_response):
        match = re.search(r'Target: (\w+)', llm_response, re.IGNORECASE)
        if match:
            return match.group(1)  # Extract the target item word
        return None  # Return None if no match is found
        
    def extract_state(self, response):
        if "State: IDENTIFICATION" in response:
            self.current_state = self.TaskState.IDENTIFICATION
            return re.sub(r"State:\s*(IDENTIFICATION)", "", response).strip()
        elif "State: MODIFICATION" in response:
            self.current_state = self.TaskState.MODIFICATION
            return re.sub(r"State:\s*(MODIFICATION)", "", response).strip()
        elif "State: DETECTION" in response:
            self.current_state = self.TaskState.DETECTION
            return re.sub(r"State:\s*(DETECTION)", "", response).strip()
        elif "State: EXECUTION" in response:
            self.current_state = self.TaskState.EXECUTION
            self.execution_flag = True
            return re.sub(r"State:\s*(EXECUTION)", "", response).strip()

    def save_convo_hist(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d")
        file_path = os.path.join(data_files_path, f"convo_{timestamp}.json")

        # Load existing history if file exists
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = []
        else:
            existing_data = []

        # Append new entries
        existing_data.extend(self.convo_hist)
        self.convo_hist = []  # Clear memory to avoid duplicates

        # Save updated history back to file
        with open(file_path, "w") as f:
            json.dump(existing_data, f, indent=4)

    def add_to_convo_hist(self, role, text):
        entry = {
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "role": role,
            "text": text
        }
        self.convo_hist.append(entry)
        self.save_convo_hist()