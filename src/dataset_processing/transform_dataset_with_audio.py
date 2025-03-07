""""
This script is used to extend the dataset with audio files.

Sample usage:
python3 -m src.dataset_processing.transform_dataset_with_audio --base_path /Users/yavuz/data --num_audio_files 2 --dataset_entity_count 150

For 5 sec data, we have 320KB output. For 3 sec, 192KB.
Currently no batching. And we do not extract embeddings either.
"""
import argparse
import os
import pandas as pd
import torch
from diffusers import AudioLDM2Pipeline
import scipy
from datetime import datetime


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
        '--num_audio_files',
        type=int,
        default=-1,
        help="Number entities to generate audio files for. (default: -1 which is all entities)"
    )
    return parser.parse_args()


def get_audio_pipe():
    repo_id = "cvssp/audioldm2"
    pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float32)

    if torch.cuda.is_available():
        pipe = pipe.to("cuda")

    return pipe


def generate_audio_from_text(text: str, pipe: AudioLDM2Pipeline):
    prompt = "Sound associateed with: " + text
    negative_prompt = "Low quality."

    output = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=10, # original was 200
        audio_length_in_s=3.0, #original was 5.0
        num_waveforms_per_prompt=2, #original was 3
    )
    return output


def main():
    args = parse_arguments()
    dataset_base_path = os.path.join(args.base_path, f"LAION-{args.dataset_entity_count}")
    num_audio_files = args.num_audio_files

    # load metadata containing the text
    metadata = pd.read_parquet(os.path.join(dataset_base_path, "metadata.parquet"))

    if num_audio_files == -1:
        num_audio_files = metadata.shape[0]
    print(f"Loaded {metadata.shape[0]} entity texts from {dataset_base_path}")

    assert num_audio_files <= metadata.shape[
        0], f"Error: num_audio_files ({num_audio_files}) is greater than the number of entities in the dataset ({metadata.shape[0]})"


    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    audio_folder = os.path.join(dataset_base_path, "audio", current_time)
    if not os.path.exists(audio_folder):
        os.makedirs(audio_folder)

    pipe = get_audio_pipe()

    for i in range(0, num_audio_files):
        text = metadata["TEXT"].iloc[i]
        print(f"Generating audio for entity {i}: {text}")
        output = generate_audio_from_text(text, pipe)
        audio = output.audios[0] # select the first waveform which is the highest scored

        audio_path = os.path.join(audio_folder, f"{i}.wav")
        scipy.io.wavfile.write(audio_path, 16000, audio)


if __name__ == "__main__":
    main()
