""""
This script is used to extend the dataset with video files, from text and image.
Sample:
python3 -m src.dataset_processing.transform_dataset_with_video --dataset_entity_count 150 --base_path /Users/yavuz/data --num_video_files 2
"""

import argparse
import os
import pandas as pd
import torch
from datetime import datetime
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image


from diffusers import I2VGenXLPipeline
from diffusers.utils import export_to_gif, load_image


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Extend dataset containig.")
    parser.add_argument(
        '--dataset_entity_count',
        type=int,
        default=100,
        help="Number of initial entities in the dataset, which indicates which folder to use e.g. LAION-100 (default: 100)"
    )
    parser.add_argument(
        '--base_path',
        type=str,
        default="data/",
        help="Base path for dataset files."
    )
    parser.add_argument(
        '--num_video_files',
        type=int,
        default=-1,
        help="Number of entities to generate video files for. (default: -1 which is all entities)"
    )
    return parser.parse_args()


def get_image_path(vector_id: int, images_path: str):
    """
    Given a vector id and base images path (IMAGES_PATH), returns the image.
    """
    shard = str(vector_id // 10000).zfill(5)
    index = str(vector_id % 10000).zfill(4)
    image_path = f"{images_path}/{shard}/{shard}{index}.jpg"
    return image_path


def main():
    args = parse_arguments()
    dataset_base_path = os.path.join(args.base_path, f"LAION-{args.dataset_entity_count}")
    num_video_files = args.num_video_files

    # load metadata containing the text
    metadata = pd.read_parquet(os.path.join(dataset_base_path, "metadata.parquet"))
    images_path = os.path.join(dataset_base_path, "images")

    if num_video_files == -1:
        num_video_files = metadata.shape[0]
    print(f"Loaded {metadata.shape[0]} entity texts from {dataset_base_path}")

    assert num_video_files <= metadata.shape[
        0], f"Error: num_video_files ({num_video_files}) is greater than the number of entities in the dataset ({metadata.shape[0]})"


    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    video_folder = os.path.join(dataset_base_path, "video", current_time)
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)



    if torch.cuda.is_available():
        pipeline = I2VGenXLPipeline.from_pretrained(
            "ali-vilab/i2vgen-xl", torch_dtype=torch.float16, variant="fp16"
        )
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = I2VGenXLPipeline.from_pretrained(
            "ali-vilab/i2vgen-xl", torch_dtype=torch.float32 #variant=fp32 modeling file is noa available
        )

    print("Loaded pipeline")

    for i in range(0, num_video_files):
        text = metadata["TEXT"].iloc[i]
        image_path = get_image_path(i, images_path)
        image = load_image(image_path)

        print(f"Generating video for entity {i}: {text}")

        prompt = "Video associated with: " + text
        negative_prompt = "Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms"

        generator = torch.manual_seed(8888)

        frames = pipeline(
            prompt=prompt,
            image=image,
            num_inference_steps=2, # original was 50
            negative_prompt=negative_prompt,
            guidance_scale=7.5, # original example was 9.0, default is 7.5
            # "A higher guidance scale value encourages the model to generate images closely linked to the text prompt at the expense of lower image quality. Guidance scale is enabled when guidance_scale > 1.
            generator=generator
        ).frames[0]

        video_path = os.path.join(video_folder, f"{i}.gif")
        export_to_gif(frames, video_path)

if __name__ == "__main__":
    main()
