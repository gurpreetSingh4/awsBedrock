import json
import base64
import os
import streamlit as st
import sys
import io

# External dependencies
import boto3
from PIL import Image
import botocore

prompt_data = """
give hd picture of lord shiv with nandi
"""
def generate_image(prompt_data):
    # prompt_template in dictionary format
    prompt_template={"text":prompt_data}

    img_config={
        "cfgScale": 7.5,
        "seed": 42,
        "quality":"standard",
        "width":1024,
        "height":1024,
        "numberOfImages":1,
    }
    bedrock = boto3.client(service_name="bedrock-runtime")
    payload = {
        "textToImageParams":prompt_template,
        "taskType":"TEXT_IMAGE",
        "imageGenerationConfig":img_config,
    }
    # json.dumps(payload) is a method that converts a Python object payload into a JSON (JavaScript Object Notation) string.
    body = json.dumps(payload)

    model_id = "amazon.titan-image-generator-v1"

    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json",
    )
    # JSON string and convert it into a Python object json.loads
    response_body = json.loads(response.get("body").read())
    print(response_body)
    img1_b64 = response_body["images"][0]

    # Debug
    print(f"Output: {img1_b64[0:80]}...")

    # Save image to a file in the output directory.
    output_dir = "output"

    os.makedirs("data/titan", exist_ok=True)
    img1 = Image.open(
        io.BytesIO(
            base64.decodebytes(
                bytes(img1_b64, "utf-8")
            )
        )
    )
    img1.save(f"data/titan/image_1.png")
    return img1


def main():
    st.title("Generated Image")
    st.write("app Generates an image based on the provided text prompt.")
    prompt_data = st.text_input("Enter your text prompt here")

    if st.button("Generate Image") and prompt_data:
        image_file = generate_image(prompt_data)
        st.image(image_file,caption="Generated AI Image",use_column_width=True)
    else: 
        st.error("Please enter a text prompt.")

if __name__ == "__main__":
    main()