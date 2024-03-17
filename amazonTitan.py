import boto3
import json
import base64
import os

prompt_data = """
provide me an 4k hd image of krishna eating cake at party
"""
prompt_template={"text":prompt_data}

img_config={
    "cfgScale": 8,
    "seed": 0,
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

body = json.dumps(payload)
model_id = "amazon.titan-image-generator-v1"
response = bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json",
)

response_body = json.loads(response.get("body").read())
print(response_body)
artifact = response_body.get("artifacts")[0]
image_encoded = artifact.get("base64").encode("utf-8")
image_bytes = base64.b64decode(image_encoded)

# Save image to a file in the output directory.
output_dir = "output1"
os.makedirs(output_dir, exist_ok=True)
file_name = f"{output_dir}/generated-img1.png"
with open(file_name, "wb") as f:
    f.write(image_bytes)

