import boto3
import json
import base64
import os
import streamlit as st

prompt_data = """
provide me an hd image of monkey who angry on cat
"""
def generate_image(prompt_data):
    # prompt_template in dictionary format
    prompt_template=[{"text":prompt_data,"weight":1}]
    bedrock = boto3.client(service_name="bedrock-runtime")
    payload = {
        "text_prompts":prompt_template,
        "cfg_scale": 10,
        "seed": 0,
        "steps":50,
        "width":512,
        "height":512

    }
    # json.dumps(payload) is a method that converts a Python object payload into a JSON (JavaScript Object Notation) string.
    body = json.dumps(payload)

    model_id = "stability.stable-diffusion-xl-v0"

    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json",
    )
    # JSON string and convert it into a Python object json.loads
    response_body = json.loads(response.get("body").read())
    print(response_body)
    artifact = response_body.get("artifacts")[0]
    image_encoded = artifact.get("base64").encode("utf-8")
    image_bytes = base64.b64decode(image_encoded)

    # Save image to a file in the output directory.
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"{output_dir}/generated-img.png"
    with open(file_name, "wb") as f:
        f.write(image_bytes)

    return file_name


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