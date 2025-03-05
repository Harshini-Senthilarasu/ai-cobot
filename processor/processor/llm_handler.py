import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv(dotenv_path="/home/harshini/capstone/src/.env")
api_key = os.getenv("google_api_key")

class llmHandler:
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
            Find the target item and the goal positions in the user prompt.
            Breakdown the user prompt into steps to be taken to move the cobot arm to fulfill the user's instructions.
            At the end of breaking down the instructions, ask the user if this is breakdown is correct by saying: Reply 'yes' to proceed or 'no' to modify."
            If the user says no, then redo the steps again with any additional information given by the user.
            If the user says yes, then if the target item exists in the list of detected objects provided with a confidence score higher than 0.6, reply with the target item name in the list. 
            If the item is not present in the list, reply saying the target item name in the prompt cannot be found in the detected objects list.
            """
        )
        self.convo_hist = []
        self.chat_session = self.model.start_chat(history=self.convo_hist)

    def get_llm_response(self, user_prompt):
        self.convo_hist.append({"role": "user", "text": user_prompt})
        response = self.chat_session.send_message(user_prompt)
        self.convo_hist.append({"role": "llm", "text": response.text})
        self.get_logger().info(f"LLM response: {response.text}")
        return response.text