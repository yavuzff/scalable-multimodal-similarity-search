import gradio as gr
import pandas as pd
import os
import numpy as np

from src.load_dataset import load_vectors_from_4_modalities_dataset_base_path, load_image, load_audio
from multivec_index import ExactMultiVecIndex
from multivec_index import MultiVecHNSW
from src.embedding_generators.text_embeddings import SentenceTransformerEmbeddingGenerator
from src.embedding_generators.image_embeddings import HFImageEmbeddingGenerator
from src.embedding_generators.audio_embeddings import AudioEmbeddingGenerator
from src.embedding_generators.video_embeddings import VideoEmbeddingGenerator

# allow Gradio to access this folder for the demo
ACCESSIBLE_DATA_PATH = "/Users/yavuz/data"
USE_EMBEDDINGS_FROM_DATASET_IF_AVAILABLE = True  # if true, for the query audio/video, if the input is from the dataset, it will use the embedding from the dataset


class LargeEntityIndexWrapper:
    """
    Wrapper for the index to be called by the Gradio interface.
    """

    def __init__(self):
        self.index: ExactMultiVecIndex = None
        self.dataset_path = None
        self.dataset_metadata = None
        self.image_embedding_generator = None
        self.text_embedding_generator = None
        self.audio_embedding_generator = None
        self.video_embedding_generator = None

    def build_index(self, dataset_folder, text_weight, image_weight, audio_weight, video_weight, text_metric,
                    image_metric, audio_metric, video_metric, index_type):
        self.dataset_path = dataset_folder

        # load vectors from dataset_folder
        try:
            text_vectors, image_vectors, audio_vectors, video_vectors = load_vectors_from_4_modalities_dataset_base_path(
                dataset_folder)
        except Exception as e:
            return f"Failed to load vectors from the dataset folder: {e}"

        print(
            f"Loaded 32-bit vectors. Text vectors shape: {text_vectors.shape}. Image vectors shape: {image_vectors.shape}. "
            f"Audio vectors shape: {audio_vectors.shape}. Video vectors shape: {video_vectors.shape}.")

        # load metadata from dataset_folder
        try:
            self.dataset_metadata = pd.read_parquet(dataset_folder + "/metadata-4-modalities.parquet")
        except Exception as e:
            return "Failed to load metadata from the dataset folder. Please check the dataset path contains metadata-4-modalities.parquet: {e}"

        # initialise the embedding generators (used only for the search)
        self.text_embedding_generator = SentenceTransformerEmbeddingGenerator()
        self.image_embedding_generator = HFImageEmbeddingGenerator()
        self.audio_embedding_generator = AudioEmbeddingGenerator()
        self.video_embedding_generator = VideoEmbeddingGenerator()

        # build index
        modalities = 4
        weights = [text_weight, image_weight, audio_weight, video_weight]
        metrics = [text_metric, image_metric, audio_metric, video_metric]
        dataset = [text_vectors, image_vectors, audio_vectors, video_vectors]
        dims = [text_vectors.shape[1], image_vectors.shape[1], audio_vectors.shape[1], video_vectors.shape[1]]
        if index_type == "MultiVecHNSW":
            self.index = MultiVecHNSW(modalities, dims, metrics, weights)
        elif index_type == "ExactMultiVecIndex":
            self.index = ExactMultiVecIndex(modalities, dims, metrics,
                                            weights)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        self.index.add_entities(dataset)

        return f"Index successfully built! You can now search the index."

    def search(self, query_image, query_text, query_audio, query_video, k, search_image_weight, search_text_weight,
               search_audio_weight, search_video_weight):
        print(
            f"Searching for image: {query_image}, text: {query_text}, audio: {query_audio}, video: {query_video}, k: {k}")

        if self.index is None:
            return "Cannot search without building an index first. Please build the index.", None

        if query_image is None:
            search_image_weight = 0

        if query_text == "":
            search_text_weight = 0

        if query_audio is None:
            search_audio_weight = 0

        if query_video is None:
            search_video_weight = 0

        if search_image_weight == 0 and search_text_weight == 0 and search_audio_weight == 0 and search_video_weight == 0:
            return "Please provide at least one of an image, text, audio or video query.", None

        if k < 1:
            return "Please provide a positive integer value for k.", None

        # generate vectors for the query image and text
        dimensions = self.index.dimensions

        if query_text != "":
            query_text_vector = self.text_embedding_generator.generate_text_embeddings([query_text])[0]
        else:
            query_text_vector = [-1] * dimensions[0]

        if query_image is not None:
            query_image_vector = self.image_embedding_generator.generate_image_embeddings([query_image])[0]
        else:
            query_image_vector = [-1] * dimensions[1]

        if query_audio is not None:
            # check if the audio is part of the dataset and already has an embedding
            query_audio_file_name = query_audio.split("/")[-1]
            potential_path = self.dataset_path + "/audios/" + query_audio_file_name
            if USE_EMBEDDINGS_FROM_DATASET_IF_AVAILABLE and os.path.exists(potential_path):
                # load all vectors
                dataset_audio_vectors = np.load(self.dataset_path + "/vectors-4-modalities/audio_vectors.npy")
                query_audio_vector = dataset_audio_vectors[int(query_audio_file_name.split(".")[0])]
                print("Loaded audio embedding from dataset.")
            else:
                # load the audio file and generate the embedding
                query_audio_vector = self.audio_embedding_generator.generate_audio_embedding(query_audio)
                print("Generated audio embedding.")
        else:
            query_audio_vector = [-1] * dimensions[2]

        if query_video is not None:
            # check if the video is part of the dataset and already has an embedding
            query_video_file_name = query_video.split("/")[-1]
            potential_path = self.dataset_path + "/videos/" + query_video_file_name
            if USE_EMBEDDINGS_FROM_DATASET_IF_AVAILABLE and os.path.exists(potential_path):
                # load all vectors
                dataset_video_vectors = np.load(self.dataset_path + "/vectors-4-modalities/video_vectors.npy")
                query_video_vector = dataset_video_vectors[int(query_video_file_name.split(".")[0])]
                print("Loaded video embedding from dataset.")
            else:
                # load the video file and generate the embedding
                query_video_vector = self.video_embedding_generator.generate_video_embedding(query_video)
                print("Generated video embedding.")
        else:
            query_video_vector = [-1] * dimensions[3]

        query = [query_text_vector, query_image_vector, query_audio_vector, query_video_vector]
        # search the index
        try:
            ids = self.index.search(query, k,
                                    query_weights=[search_text_weight, search_image_weight, search_audio_weight,
                                                   search_video_weight])
        except Exception as e:
            return f"Search failed with error: {e}", None

        # retrieve images and text with id from the dataset
        entity_results = []
        # load image and text
        images_path = self.dataset_path + "/images"
        audios_path = self.dataset_path + "/audios"
        videos_path = self.dataset_path + "/videos"
        for entity_id in ids:
            image = load_image(entity_id, images_path)
            text = self.dataset_metadata["TEXT"][entity_id]
            audio = audios_path + "/" + str(entity_id) + ".wav"
            video = videos_path + "/" + str(entity_id) + ".mp4"

            entity = (entity_id, image, text, audio, video)
            entity_results.append(entity)

        return f"Search completed successfully. Returned {len(ids)} entities.", entity_results


# initialise the index wrapper that will be used through the demo
index_wrapper = LargeEntityIndexWrapper()

# main interface
with gr.Blocks(title="Multimodal Similarity Search Demo") as demo:
    gr.Markdown("# Demonstration of the Multimodal Similarity Search Framework")

    # build index section
    with gr.Tab("Build Multimodal Index"):
        gr.Markdown("### Build the index from your dataset folder")
        with gr.Row():
            dataset_folder_input = gr.Textbox(label="Dataset Folder Path",
                                              placeholder="Enter the path to your dataset folder")
            dataset_folder_input.value = "/Users/yavuz/data/LAION-150-4-modalities"
            index_type = gr.Dropdown(label="Index Type", choices=["ExactMultiVecIndex", "MultiVecHNSW"],
                                     value="ExactMultiVecIndex")

        with gr.Row():
            image_weight_slider = gr.Slider(0, 1, value=0.5, label="Image Weight")
            image_metric = gr.Dropdown(label="Image Metric", choices=["cosine", "euclidean", "manhattan"],
                                       value="cosine")
        with gr.Row():
            text_weight_slider = gr.Slider(0, 1, value=0.5, label="Text Weight")
            text_metric = gr.Dropdown(label="Text Metric", choices=["cosine", "euclidean", "manhattan"], value="cosine")

        with gr.Row():
            audio_weight_slider = gr.Slider(0, 1, value=0.5, label="Audio Weight")
            audio_metric = gr.Dropdown(label="Audio Metric", choices=["cosine", "euclidean", "manhattan"],
                                       value="cosine")
        with gr.Row():
            video_weight_slider = gr.Slider(0, 1, value=0.5, label="Video Weight")
            video_metric = gr.Dropdown(label="Video Metric", choices=["cosine", "euclidean", "manhattan"],
                                       value="cosine")

        build_button = gr.Button("Build Index")
        build_status = gr.Textbox(label="Status", placeholder="Build status will be shown here")

        build_button.click(fn=index_wrapper.build_index,
                           inputs=[dataset_folder_input, text_weight_slider, image_weight_slider, audio_weight_slider,
                                   video_weight_slider,
                                   text_metric, image_metric, audio_metric, video_metric, index_type],
                           outputs=build_status)

    # search index section
    with gr.Tab("Search Multimodal Index"):
        gr.Markdown("### Search the index")
        with gr.Row():
            query_image_input = gr.Image(label="Query Image", type="filepath")
            query_text_input = gr.Textbox(label="Query Text", placeholder="Enter text query")

        with gr.Row():
            query_audio_input = gr.Audio(label="Query Audio", type="filepath")
            query_video_input = gr.Video(label="Query Video", format="mp4")

        with gr.Row():
            search_image_weight_slider = gr.Slider(0, 1, value=image_weight_slider.value,
                                                   label="Image Weight for Search")
            search_text_weight_slider = gr.Slider(0, 1, value=text_weight_slider.value, label="Text Weight for Search")
            search_audio_weight_slider = gr.Slider(0, 1, value=audio_weight_slider.value,
                                                   label="Audio Weight for Search")
            search_video_weight_slider = gr.Slider(0, 1, value=video_weight_slider.value,
                                                   label="Video Weight for Search")

            # add event listeners to update the search sliders whenever the index build sliders are updated
            image_weight_slider.change(
                fn=lambda x: x,
                inputs=image_weight_slider,
                outputs=search_image_weight_slider
            )
            text_weight_slider.change(
                fn=lambda x: x,
                inputs=text_weight_slider,
                outputs=search_text_weight_slider
            )
            audio_weight_slider.change(
                fn=lambda x: x,
                inputs=image_weight_slider,
                outputs=search_audio_weight_slider
            )
            video_weight_slider.change(
                fn=lambda x: x,
                inputs=image_weight_slider,
                outputs=search_video_weight_slider
            )

        # k input and search button
        k_input = gr.Number(value=5, label="Number of Neighbours (k)", precision=0)
        search_button = gr.Button("Search")

        # output components
        search_status = gr.Textbox(label="Search Status")

        MAX_DISPLAYED_ENTITIES = 10
        result_containers = []
        for i in range(MAX_DISPLAYED_ENTITIES):
            with gr.Column(visible=False) as container:
                title = gr.Markdown(f"### Entity _")
                with gr.Row():
                    img = gr.Image(label="Image", type="pil")
                    txt = gr.Textbox(label="Text", interactive=False)
                with gr.Row():
                    audio = gr.Audio(label="Audio", type="filepath", interactive=False)
                    video = gr.Video(label="Video", format="mp4", interactive=False)
            result_containers.append((container, title, img, txt, audio, video))


    # function to update the containers with search results
    def display_search_results(query_image_path, query_text, query_audio_path, query_video_path, k, search_image_weight,
                               search_text_weight, search_audio_weight, search_video_weight):
        status, entity_results = index_wrapper.search(query_image_path, query_text, query_audio_path, query_video_path,
                                                      k, search_image_weight, search_text_weight, search_audio_weight,
                                                      search_video_weight)
        updates = [status]

        # if search fails, display only the status message
        if entity_results is None:
            for _ in range(MAX_DISPLAYED_ENTITIES):
                updates.extend([gr.update(visible=False, ), None, None, "", None, None])
            return updates

        # update the rows of the UI with search results
        k = len(entity_results)
        for i in range(MAX_DISPLAYED_ENTITIES):
            if i < k:
                # update the container to show the image, text, audio and video
                updates.extend([
                    gr.update(visible=True),
                    gr.Markdown(f"### Entity {entity_results[i][0]}"),
                    entity_results[i][1],
                    entity_results[i][2],
                    entity_results[i][3],
                    entity_results[i][4],
                ])
            else:
                # hide the row and clear image, text, audio and video for unused rows
                updates.extend([gr.update(visible=False), None, None, "", None, None])
        return updates


    # for each container we have 5 components: container update, image, text, audio and video
    outputs = [search_status] + [comp for container in result_containers for comp in container]
    search_button.click(fn=display_search_results,
                        inputs=[query_image_input, query_text_input, query_audio_input, query_video_input, k_input,
                                search_image_weight_slider, search_text_weight_slider, search_audio_weight_slider,
                                search_video_weight_slider],
                        outputs=outputs)

demo.launch(allowed_paths=[ACCESSIBLE_DATA_PATH])
