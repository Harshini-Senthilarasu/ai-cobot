'''
Brief:      This file contains the helper class llmHandler
            - sets up the Gemini model
            - gets response from the llm
            - finds the target objects, TaskState and CobotAction
            - saves the conversion history 
'''
import os
from dotenv import load_dotenv
import google.generativeai as genai
from enum import Enum
import json
import datetime
import re
from geometry_msgs.msg import Point

load_dotenv(dotenv_path="/home/harshini/capstone/src/.env")
api_key = os.getenv("google_api_key")
data_files_path = os.getenv("convo_log_path")

class llmHandler:
    class TaskState(Enum):
        IDENTIFICATION  = "IDENTIFICATION"
        DETECTION       = "DETECTION"
        BREAKDOWN       = "BREAKDOWN"
        MODIFICATION    = "MODIFICATION"
        EXECUTION       = "EXECUTION"
        STANDBY         = "STANDBY"

    class CobotAction(Enum):
        MOVE    = "move"
        CLOSE   = "close"
        OPEN    = "open"
        HOME    = "home"
        STANDBY = "standby"

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
            1. IDENTIFICATION - Extract the target item in the user prompt. If the target item is found in the user prompt, change .
            2. DETECTION - Analyse the detected object list to find the closest possible object to the target item based on the label and the confidence score. If target item found, move on to BREAKDOWN State.
            3. BREAKDOWN - Create a list of steps the cobot arm needs to perform to fulfill the user's instructions. Example "1. Move to position (x,y,z) mm 2. Close gripper" and such. Ask the user for confirmation: Reply 'yes' to proceed or 'no' to modify.
            4. MODIFICATION - If the user says no in BREAKDOWN State, modify the task breakdown. Request for more details if required.
            5. EXECUTION - If the user says yes in BREAKDOWN Sate, start the cobot step with the first step (1) in the task breakdown. If the response is returned as success, continue with the next step. If the response is not success, indicate failure and change cobot step to 0. Once all steps in breakdown are completed, return cobot step value to 0 and wait for next task
            6. STANDBY - Once all the steps in EXECUTION have been successfully completed, send a message saying "Task completed.", then wait for the next prompt from the user. If the user prompts saying the cobot is returning home, wait in this state until the user prompts that the cobot has returned home. 
            The starting State is IDENTIFICATION. Change the State if/when required.
            Rules:
            1. Ask: "Does this breakdown look correct? Reply 'yes' or 'no'."
            2. Indicate target item found in user prompt as 'Target Item from user: "TARGET_ITEM_NAME"'
            3. Indicate target item found in detected object list as 'Target Item in list: "TARGET_ITEM_NAME"'. If target item cannot be found in detected object list, indicate as "Target Item in list: NOT_FOUND" and request to resend detected object list
            4. Initial Cobot Step is 0. In Step 4, indicate the cobot step value as "Cobot Step: COBOT_STEP" and "Total Cobot Steps: COBOT_STEPS"
            5. Ask: "Can I execute the movment?" before starting EXECUTION State. If user says no, return to IDENTIFICATION State.
            6. Always indicate what State at the end of the response as such: 'State: IDENTIFICATION'
            7. To close the gripper, say "Close gripper". To open gripper, say "Open gripper". To move a position, say "Move to position (x,y,z) mm" using the values from the detected object list
            """
        )

        # Variables
        self.convo_hist = []
        self.chat_session = self.model.start_chat(history=self.convo_hist)
        os.makedirs(data_files_path, exist_ok=True)
        self.current_state = self.TaskState.IDENTIFICATION
        self.target_item_user = None
        self.target_item_list = None
        self.cobot_step = None
        self.total_cobot_steps = None
        self.gripper_state = 0
        self.position_from_llm = Point()
        self.cobot_state = None

        self.logger.info("LLM Handler started")
    
    '''
    Function to get the response from llm
    '''
    def get_llm_response(self, user_prompt):
        self.add_to_convo_hist("user", user_prompt)
        try: 
            response = self.chat_session.send_message(user_prompt)
        except genai.errors.APIError as e:
            self.logger.info(f"LLM Handler: Error in LLM")
        self.add_to_convo_hist("llm", response.text)
        llm_response = self.extract_state(response.text)
        self.extract_pattern(llm_response)
        # self.logger.info(f"LLM Handler: Current State is {self.current_state}")
        # self.logger.info(f"LLM response: {llm_response}")
        return llm_response

    '''
    Get the target objects and CobotAction
    '''
    def extract_pattern(self, llm_response):
        if llm_response is None:
            self.logger.info("LLM Handler: llm response is none")
            return

        # patterns
        target_item_user_pattern = r'target item from user:\s*"([^"]+)"'
        target_item_list_pattern = r'target Item in list:\s*"([^"]+)"'
        cobot_step_pattern = r'cobot step:\s*(\d+)'
        total_cobot_steps_pattern = r'total cobot steps:\s*(\d+)'
        position = r'\(\s*(-?\d+\.\d+|-?\d+),\s*(-?\d+\.\d+|-?\d+),\s*(-?\d+\.\d+|-?\d+)\s*\)'

        llm_response = llm_response.lower()
        llm_response = llm_response.replace("\n", " ").strip()
        # self.logger.info(f"LLM Handler: {llm_response}")

        if self.current_state == self.TaskState.IDENTIFICATION:
            self.target_item_user = self.match_pattern(llm_response, target_item_user_pattern)
            self.logger.info(f"LLM Handler: Target item from user is {self.target_item_user}")

        elif self.current_state == self.TaskState.DETECTION:
            self.target_item_list = self.match_pattern(llm_response, target_item_list_pattern)
            # If "Target Item in list" is not found, set to None
            if self.target_item_list == "not_found":
                self.target_item_list = None
            self.logger.info(f"LLM Handler: Target item in list is {self.target_item_list}")

        elif self.current_state == self.TaskState.EXECUTION:
            # self.cobot_step = self.match_pattern(llm_response, cobot_step_pattern)
            # self.total_cobot_steps = self.match_pattern(llm_response, total_cobot_steps_pattern)
            # self.logger.info(f"LLM Handler: Cobot Step: {self.cobot_step}/{self.total_cobot_steps}")

            if "position" in llm_response:
                self.position_from_llm = self.match_pattern(llm_response, position, is_position=True)
                # if self.position_from_llm:
                self.cobot_state = self.CobotAction.MOVE
                self.logger.info(f"LLM Handler: Position {self.position_from_llm}")

            elif "gripper" in llm_response:
                if "open" in llm_response:
                    self.cobot_state = self.CobotAction.OPEN
                elif "close" in llm_response:
                    self.cobot_state = self.CobotAction.CLOSE               

    '''
    Function to match the patterns
    '''
    def match_pattern(self, text, pattern, is_position=False):
        match = re.search(pattern, text)
        if match:
            if is_position:
                x, y, z = map(float, match.groups())
                position_msg = Point()
                position_msg.x = x
                position_msg.y = y
                position_msg.z = z
                return position_msg  # Return a Point message
            return match.group(1)  # Extract a regular match
        return None

    '''
    Function to extract the states from the response
    '''
    def extract_state(self, response):
        if "State: IDENTIFICATION" in response:
            self.current_state = self.TaskState.IDENTIFICATION
            return re.sub(r"State:\s*(IDENTIFICATION)", "", response).strip()
        elif "State: DETECTION" in response:
            self.current_state = self.TaskState.DETECTION
            return re.sub(r"State:\s*(DETECTION)", "", response).strip()
        elif "State: BREAKDOWN" in response:
            self.current_state = self.TaskState.BREAKDOWN
            return re.sub(r"State:\s*(BREAKDOWN)", "", response).strip()
        elif "State: MODIFICATION" in response:
            self.current_state = self.TaskState.MODIFICATION
            return re.sub(r"State:\s*(MODIFICATION)", "", response).strip()
        elif "State: EXECUTION" in response:
            self.current_state = self.TaskState.EXECUTION
            return re.sub(r"State:\s*(EXECUTION)", "", response).strip()
        elif "State: STANDBY" in response:
            self.current_state = self.TaskState.STANDBY
            return re.sub(r"State:\s*(STANDBY)", "", response).strip()

    '''
    Function to save the conversation history into file
    '''
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