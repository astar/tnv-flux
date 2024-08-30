import argparse
from .config import load_settings
from .image_generator import generate_and_save_image

def main():
    parser = argparse.ArgumentParser(description="Generate an image using Flux model")
    parser.add_argument("prompt", help="Additional prompt text to be added to the system prompt")
    args = parser.parse_args()

    settings = load_settings()

    full_prompt = f"{settings['system_prompt']} {args.prompt}"

    output_url = generate_and_save_image(settings, full_prompt)

    print(f"Image generated and saved: {output_url}")

if __name__ == "__main__":
    main()