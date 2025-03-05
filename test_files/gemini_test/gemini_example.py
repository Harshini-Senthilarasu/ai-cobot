import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("api_key")
genai.configure(api_key=api_key)

def upload_to_gemini(path, mime_type=None):
  """Uploads the given file to Gemini.

  See https://ai.google.dev/gemini-api/docs/prompting_with_media
  """
  file = genai.upload_file(path, mime_type=mime_type)
  print(f"Uploaded file '{file.display_name}' as: {file.uri}")
  return file

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-2.0-flash",
  generation_config=generation_config,
  system_instruction="You are a object detector. \nIdentify the target object described in the prompt in the image provided.\nFind the coordinates of the centroid of the target object and its bounding box assuming the origin (0,0) is the top left corner of the image.\nProvide the 5 sets of coordinates in the following format: \n\"Centroid: [x, y]\"\n\"Top Left of bounding box: [x, y]\"\n\"Top Right of bounding box: [x, y]\"\n \"Bottom Left of bounding box: [x, y]\"\n\"Bottom Right of bounding box: [x, y]\"\nIf the object is not found, return an empty response \"{}\".",
)

# TODO Make these files available on the local file system
# You may need to update the file paths
# files = [
#   upload_to_gemini("", mime_type="image/png"),
#   upload_to_gemini("", mime_type="image/png"),
# ]

chat_session = model.start_chat(history=[])

response = chat_session.send_message("short introduction of AI")

print(response.text)