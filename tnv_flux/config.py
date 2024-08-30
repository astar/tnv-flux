import os
from dotenv import load_dotenv
from typing import TypedDict, Literal

class FluxSettings(TypedDict):
    service: Literal["replicate", "huggingface"]
    replicate_model: str
    huggingface_model: str
    system_prompt: str
    replicate_api_key: str

def load_settings() -> FluxSettings:
    load_dotenv()

    return FluxSettings(
        service=os.getenv('FLUX_SERVICE', 'replicate'),
        replicate_model=os.getenv('FLUX_REPLICATE_MODEL', 'adirik/flux-cinestill:216a43b9975de9768114644bbf8cd0cba54a923c6d0f65adceaccfc9383a938f'),
        huggingface_model=os.getenv('FLUX_HUGGINGFACE_MODEL', 'black-forest-labs/FLUX.1-schnell'),
        system_prompt=os.getenv('FLUX_SYSTEM_PROMPT', 'CNSTLL, Road trip, view through car window of desert highway'),
        replicate_api_key=os.getenv('REPLICATE_API_KEY', '')
    )