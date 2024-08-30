import replicate
import requests
import os
import shutil
from gradio_client import Client

def generate_and_save_image(settings: dict, prompt: str) -> str:
    if settings['service'] == 'replicate':
        return generate_replicate_image(settings['replicate_model'], prompt, settings['replicate_api_key'])
    elif settings['service'] == 'huggingface':
        return generate_huggingface_image(settings['huggingface_model'], prompt)
    else:
        raise ValueError(f"Invalid service: {settings['service']}")

def generate_replicate_image(model: str, prompt: str, api_key: str) -> str:
    os.environ['REPLICATE_API_TOKEN'] = api_key

    output = replicate.run(
        model,
        input={
            "model": "dev",
            "prompt": prompt,
            "lora_scale": 0.6,
            "num_outputs": 1,
            "aspect_ratio": "1:1",
            "output_format": "png",
            "guidance_scale": 3.5,
            "output_quality": 80,
            "extra_lora_scale": 0.8,
            "num_inference_steps": 28
        }
    )

    image_url = output[0]
    local_file_path = "generated_image.png"

    response = requests.get(image_url)
    if response.status_code == 200:
        with open(local_file_path, 'wb') as file:
            file.write(response.content)
        print(f"Image saved locally as {local_file_path}")
    else:
        print(f"Failed to download the image. Status code: {response.status_code}")

    return local_file_path

def generate_huggingface_image(model: str, prompt: str) -> str:
    client = Client(model)
    result = client.predict(
        prompt=prompt,
        seed=0,
        randomize_seed=True,
        width=1024,
        height=1024,
        num_inference_steps=28,
        api_name="/infer"
    )
    file_path = result[0]
    local_file_path = "generated_image.png"
    shutil.copy(file_path, local_file_path)
    return local_file_path