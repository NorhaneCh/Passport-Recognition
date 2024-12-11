import json

import ollama
import streamlit as st
from PIL import Image

# Streamlit app layout
st.title("Passport Recognition with Ollama Vision")

import os

# Get the absolute path of the prompt.txt file
script_dir = os.path.dirname(os.path.abspath(__file__))
prompt_file_path = os.path.join(script_dir, 'prompt.txt')

# Read the prompt content from a file
try:
    with open(prompt_file_path, 'r', encoding='utf-8') as file:
        prompt_content = file.read()
    st.write("Prompt content loaded successfully.")
except FileNotFoundError:
    st.write(f"Error: {prompt_file_path} not found.")

# Upload image
uploaded_file = st.file_uploader(
    "Upload an image of a passport for data extraction", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Open the uploaded image using PIL
        image = Image.open(uploaded_file)

        # Display the original uploaded image
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Resize the image
        base_width = 800
        w_percent = (base_width / float(image.size[0]))
        new_height = int((float(image.size[1]) * float(w_percent)))

        # Resize the image with the LANCZOS filter (high-quality resampling)
        resized_image = image.resize(
            (base_width, new_height), Image.Resampling.LANCZOS)

        # Save the resized image to pass it to the Ollama API
        resized_image.save("resized_image.jpg")

        # Call Ollama API with the resized image
        response = ollama.chat(
            model='llama3.2-vision',
            messages=[{
                'role': 'user',
                'content': prompt_content,
                'images': ['resized_image.jpg']
            }]
        )

        # Extract the content field where the JSON is located
        assistant_content = response.message.content

        # Find the JSON portion using string manipulation
        try:
            start_idx = assistant_content.index('{')
            end_idx = assistant_content.rindex('}') + 1
            json_data = assistant_content[start_idx:end_idx]
            # Convert string to Python dictionary
            parsed_json = json.loads(json_data)

            # Display the extracted JSON in the Streamlit app
            st.subheader("Extracted JSON Data")
            st.json(parsed_json)  # Streamlit will pretty-print the JSON

        except ValueError:
            st.error("Failed to extract JSON from the response.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
