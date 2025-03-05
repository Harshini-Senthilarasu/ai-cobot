###
# Author    : Harshini Senthilarasu
# Brief     : Test Gemini API 
###
import os
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai

load_dotenv(dotenv_path="/home/harshini/capstone/src/.env")
api_key = os.getenv("google_api_key")
genai.configure(api_key=api_key)

# Set up Gemini model
model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    system_instruction=[
        "You are connected to collaborative robot."
        "Parse the instructions provided in the prompt."
        "Identify the target item and the steps to be taken to fulfill the task."
        "Breakdown the steps to be taken and respond with those steps."
        "Check if the steps are what the user wishes to do."
    ]
)

st.set_page_config(page_title="Gemini Test", layout="wide") # Streamlit page config

# Initialise chat session in Streamlit 
if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])

# Display chat history
for message in st.session_state.chat_session.history:
    with st.chat_message(translate_role_for_streamlit(message.role)):
        st.markdown(message.parts[0].text)

# Input field for user's message
user_prompt = st.chat_input("Enter message")
if user_prompt:
    # Add user's message to chat and display it
    st.chat_message("user").markdown(user_prompt)

    # Send user's message to Gemini-Pro and get the response
    gemini_response = st.session_state.chat_session.send_message(user_prompt)

    # Display Gemini-Pro's response
    with st.chat_message("assistant"):
        st.markdown(gemini_response.text)