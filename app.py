# /// script
# dependencies = [
#   "google-genai",
#   "fal-client",
#   "Pillow",
#   "ffmpeg-python",
#   "tqdm",
#   "httpx",
# ]
# ///

import argparse
import asyncio
import os
import uuid
import fal_client
import base64
from pathlib import Path
from PIL import Image
import io
import ffmpeg
from tqdm import tqdm
from datetime import datetime
import httpx


PROMPTS = {
    "up": "Gently pan the camera up, extending the image.",
    "down": "Gently pan the camera down, extending the image.",
    "left": "Gently pan the camera left, extending the image.",
    "right": "Gently pan the camera right, extending the image.",
    "rotate-left": "Gently rotate the camera counter-clockwise, extending the borders to fit the new perspective.",
    "rotate-right": "Gently rotate the camera clockwise, extending the borders to fit the new perspective.",
    "zoom-in": "Gently zoom in on the center of the image, maintaining focus and detail.",
    "zoom-out": "Gently zoom out from the image, revealing more of the surrounding scene.",
    "future": "Show this scene one second in the future",
    "past": "Show this scene one second in the past",
    "funny": "Subtly alter this image by replacing one or two details with something unexpected and funny.",
    "serious": "Subtly alter this image by replacing one or two details with something more serious, meaningful, or thought-provoking.",
    "dramatic": "Subtly enhance the drama and intensity of this scene. Adjust lighting to be more cinematic, deepen shadows, or add atmospheric elements like mist or dramatic sky. Keep changes photorealistic and well-integrated.",
    "peaceful": "Transform this scene to be more peaceful and serene. Soften harsh elements, add calming details like gentle lighting or natural elements. Keep changes subtle and photorealistic.",
    "vintage": "Apply a subtle vintage aesthetic to this image. Add slight film grain, adjust colors to warmer or cooler vintage tones, and create a nostalgic atmosphere while maintaining photorealism.",
    "futuristic": "Subtly modernize or add futuristic elements to this scene. Replace one or two objects with sleek, high-tech alternatives. Keep changes minimal, well-integrated, and photorealistic.",
    "nature": "Subtly introduce natural elements into this scene. Add plants, natural lighting, or organic textures. Keep changes small and seamlessly integrated with photorealistic quality.",
    "urban": "Subtly add urban elements to this scene. Introduce architectural details, city textures, or modern infrastructure. Keep changes minimal and photorealistic.",
    "minimalist": "Simplify this scene with minimalist aesthetics. Remove or tone down one or two distracting elements, create cleaner compositions, and emphasize negative space. Keep it photorealistic.",
    "crowded": "Subtly add more people or objects to make this scene feel more populated or busy. Keep additions natural, well-integrated, and photorealistic.",
    "empty": "Subtly remove one or two people or objects to make this scene feel more spacious or isolated. Keep the result natural and photorealistic.",
}
ENHANCE_PROMPT = "Enhance image resolution and quality with professional-grade restoration: Increase image resolution by 2-4x while preserving fine details and textures. Remove compression artifacts, noise, pixelation, and blur. Correct color balance, contrast, and exposure inconsistencies. Sharpen edges and text without over-sharpening. Maintain natural skin tones and realistic lighting. Fix any distortions, chromatic aberration, or lens artifacts. Ensure the final result appears crisp, clean, and professionally processed."


def get_prompt_for_mode(mode: str, custom_prompt: str = None) -> str:
    """
    Generates a prompt based on the specified generation mode.
    """
    if mode == "custom":
        if not custom_prompt:
            raise ValueError("Custom prompt is required for 'custom' mode.")
        return custom_prompt

    prompt = PROMPTS.get(mode)
    if not prompt:
        raise ValueError(f"Unknown mode: {mode}")

    return prompt


def fetch_image_from_url_base64(url: str) -> str:
    """
    Fetches an image from a URL and returns it as a base64 encoded string.
    """
    response = httpx.get(url)
    base64_string = base64.b64encode(response.content).decode("utf-8")
    return f"data:{response.headers['Content-Type']};base64,{base64_string}"


def image_to_data_uri(file_path: str) -> str:
    """
    Reads an image file and returns it as a data URI.
    """
    try:
        image = Image.open(file_path)

        # Convert image to bytes
        output_buffer = io.BytesIO()
        image.save(output_buffer, format="PNG")
        image_data = output_buffer.getvalue()

        encoded_string = base64.b64encode(image_data).decode("utf-8")
        return f"data:image/png;base64,{encoded_string}"
    except Exception as e:
        raise ValueError(f"File '{file_path}' is not a valid image file: {e}")


def rescale_image(data_uri: str, max_dimension: int = 2048) -> str:
    """
    Rescales an image from a data URI to a max dimension, keeping aspect ratio.
    """
    _, encoded = data_uri.split(",", 1)
    image_data = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(image_data))

    # # Only rescale if the image is larger than max_dimension
    # width, height = image.size
    # if width <= max_dimension and height <= max_dimension:
    #     # Return original data URI if no rescaling needed (avoid re-encoding)
    #     return data_uri

    # Use high-quality resampling and preserve quality
    image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
    output_buffer = io.BytesIO()

    image.save(output_buffer, format="PNG", optimize=False)

    encoded_string = base64.b64encode(output_buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded_string}"


async def edit_image_fal(
    prompt: str,
    image_url: str,
    model: str,
    size: tuple[int, int] = None,
    use_fal_storage: bool = False,
) -> str | None:
    """
    Edits an image using a specified Fal AI model.

    Args:
        prompt: The editing prompt
        image_url: The image URL (data URI, HTTP URL, etc.)
        model: The Fal AI model to use
        use_fal_storage: Whether to use fal storage for uploading images

    Returns:
        The edited image as a data URI
    """
    # Upload to fal storage if requested and image is a data URI
    if use_fal_storage and image_url.startswith("data:"):
        # Convert data URI to file and upload
        _, encoded = image_url.split(",", 1)
        image_data = base64.b64decode(encoded)

        # Save temporarily to upload
        temp_path = Path(f"/tmp/temp_upload_{uuid.uuid4().hex}.png")
        with open(temp_path, "wb") as f:
            f.write(image_data)

        try:
            image_url = await fal_client.upload_file_async(str(temp_path))
        finally:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()

    # image_url = rescale_image(image_url)
    try:
        arguments = {
            "prompt": prompt,
            "image_urls": [image_url],
            "num_images": 1,
            "output_format": "png",
            "sync_mode": True,
        }
        if model == "fal-ai/gpt-image-1/edit-image/byok":
            arguments["openai_api_key"] = os.environ["OPENAI_KEY"]
            arguments["quality"] = "high"
            arguments["input_fidelity"] = "high"

        if (
            model == "fal-ai/bytedance/seedream/v4/edit"
            or model == "fal-ai/qwen-image-edit-plus"
        ):
            if size:
                arguments["image_size"] = {
                    "width": size[0],
                    "height": size[1],
                }
            else:
                arguments["image_size"] = {
                    "width": 2048,
                    "height": 2048,
                }

        handler = await fal_client.submit_async(
            model,
            arguments=arguments,
        )

        result = await handler.get()

        if "images" in result and len(result["images"]) > 0:
            new_image = result["images"][0]
            url = new_image["url"]
            # check if urls is http or https , fetch it and convert to data uri
            if url.startswith("http") or url.startswith("https"):
                return fetch_image_from_url_base64(url)
            else:
                return url
        else:
            print(f"❌ Fal AI API Error: {result}")
            return None
    except Exception as e:
        print(f"❌ An exception occurred with Fal AI: {e}")
        return None


def save_data_uri_as_image(data_uri: str, output_path: Path) -> int:
    """
    Saves a data URI as a PNG image file.

    Args:
        data_uri: The data URI containing the image
        output_path: Path where to save the image (without extension)

    Returns:
        file_size_bytes
    """
    _, encoded = data_uri.split(",", 1)
    image_data = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(image_data))
    file_path = output_path.with_suffix(".png")
    image.save(file_path, format="PNG", optimize=False)
    return len(image_data)


def generate_video(images_dir: str, output_file: str, frame_rate: int = 12):
    """
    Generates a video from a directory of images.
    """
    import glob

    frame_files = glob.glob(f"{images_dir}/frame_*.png")
    num_frames = len(frame_files)

    if num_frames == 0:
        print(f"❌ No frames found in {images_dir}")
        return

    print(f"🎥 Generating video from {num_frames} frames...")

    try:
        (
            ffmpeg.input(
                f"{images_dir}/frame_*.png",
                pattern_type="glob",
                framerate=frame_rate,
            )
            .output(output_file, vcodec="libx264", pix_fmt="yuv420p")
            .run(overwrite_output=True, quiet=True)
        )

        file_size = Path(output_file).stat().st_size / (1024 * 1024)  # MB
        print(f"✅ Video saved to {output_file} ({file_size:.1f} MB)")

    except ffmpeg.Error as e:
        print(
            f"❌ Error generating video: {e.stderr.decode('utf8') if e.stderr else str(e)}"
        )
        raise


async def main():
    """
    Main function to run the image generation process.
    """
    parser = argparse.ArgumentParser(
        description="Generate image animations using Fal AI."
    )
    parser.add_argument(
        "--image_url",
        type=str,
        required=True,
        help="URL or local path of the initial image.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=list(PROMPTS.keys()),
        help="Generation mode.",
    )
    parser.add_argument(
        "--num_frames", type=int, default=10, help="Number of frames to generate."
    )
    parser.add_argument(
        "--custom_prompt", type=str, help="Custom prompt for 'custom' mode."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save generated frames.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="fal-ai/nano-banana/edit",
        choices=[
            "fal-ai/nano-banana/edit",
            "fal-ai/bytedance/seedream/v4/edit",
            "fal-ai/gpt-image-1/edit-image/byok",
            "fal-ai/qwen-image-edit-plus",
        ],
        help="Model to use for image generation (fal-ai/nano-banana/edit, fal-ai/bytedance/seedream/v4/edit, fal-ai/gpt-image-1/edit-image/byok).",
    )
    parser.add_argument(
        "--no_fal_storage",
        action="store_true",
        default=False,
        help="Do not upload images to fal storage, use data URIs instead.",
    )
    parser.add_argument(
        "--enhance",
        action="store_true",
        default=False,
        help="Enable enhancement step to improve image quality between frames.",
    )

    args = parser.parse_args()

    if "FAL_KEY" not in os.environ:
        print("Error: FAL_KEY environment variable not set.")
        return
    print("Fal AI client initialized")

    # Handle image input: URL, local file path, or Data URI
    input_image = args.image_url
    if Path(input_image).exists():
        print(f"Encoding local file: {input_image}")
        try:
            current_image_url = image_to_data_uri(input_image)
            print("File encoded as data URI.")
        except (ValueError, IOError) as e:
            print(f"Error processing file: {e}")
            return
    elif (
        input_image.startswith("http://")
        or input_image.startswith("https://")
        or input_image.startswith("data:")
    ):
        current_image_url = input_image
    else:
        print(
            f"Error: Input image '{input_image}' is not a valid URL, local file path, or Data URI."
        )
        return

    input_image_size = Image.open(input_image).size

    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_id = uuid.uuid4().hex[:2]
    model_name_for_path = args.model.replace("/", "_")
    output_dir = (
        base_output_dir
        / f"run-{model_name_for_path}-{args.mode}-{timestamp}-{run_id}"
        / "images"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving frames to {output_dir}")

    # Save current image as frame 000
    file_size = save_data_uri_as_image(current_image_url, output_dir / "frame_000")
    print(f"✅ Saved initial frame as frame_000.png ({file_size // 1024}KB)")

    try:
        prompt = get_prompt_for_mode(args.mode, args.custom_prompt)
    except ValueError as e:
        print(f"Error: {e}")
        return

    successful_frames = 0

    for i in tqdm(range(args.num_frames), desc="🎬 Generating frames", unit="frame"):
        new_image_url = await edit_image_fal(
            prompt,
            current_image_url,
            args.model,
            input_image_size,
            not args.no_fal_storage,
        )

        if new_image_url is None:
            print(f"❌ Skipping frame {i + 1} due to an API error.")
            continue

        # enhance image quality if enabled
        if args.enhance:
            enhanced_image_url = await edit_image_fal(
                ENHANCE_PROMPT,
                new_image_url,
                args.model,
                input_image_size,
                not args.no_fal_storage,
            )
            if enhanced_image_url is not None:
                new_image_url = enhanced_image_url

        current_image_url = new_image_url

        try:
            file_size = save_data_uri_as_image(
                current_image_url, output_dir / f"frame_{i + 1:03d}"
            )
            successful_frames += 1
        except Exception as e:
            print(f"❌ Error saving frame {i + 1}: {e}")
            break

    # Show final statistics
    if successful_frames > 0:
        print(f"✅ Successfully generated {successful_frames}/{args.num_frames} frames")

    video_output_file = output_dir.parent / "animation.mp4"
    generate_video(str(output_dir), str(video_output_file))


if __name__ == "__main__":
    asyncio.run(main())
