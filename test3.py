import replicate
import requests

# Make the prediction with replicate
output = replicate.run(
    "adirik/flux-cinestill:216a43b9975de9768114644bbf8cd0cba54a923c6d0f65adceaccfc9383a938f",
    input={
        "model": "dev",
        "prompt": "CNSTLL, Road trip, view through car window of desert highway\n",
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

# Print the output URL
print(output)

# The output is a list of URLs. Get the first URL.
image_url = output[0]

# Define the local file path where you want to save the image.
local_file_path = "generated_image.png"

# Download and save the image.
response = requests.get(image_url)
if response.status_code == 200:
    with open(local_file_path, 'wb') as file:
        file.write(response.content)
    print(f"Image saved locally as {local_file_path}")
else:
    print(f"Failed to download the image. Status code: {response.status_code}")