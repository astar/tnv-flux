
import shutil
from gradio_client import Client

client = Client("black-forest-labs/FLUX.1-schnell")
result = client.predict(
		prompt="Hello!!",
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