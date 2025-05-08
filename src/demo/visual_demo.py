import gradio as gr
import pandas as pd

from src.common.load_dataset import load_vectors_from_dataset_base_path, load_image
from multivec_index import ExactMultiVecIndex
from multivec_index import MultiVecHNSW
from src.embedding_generators.text_embeddings import SentenceTransformerEmbeddingGenerator
from src.embedding_generators.image_embeddings import HFImageEmbeddingGenerator


class IndexWrapper:
    """
    Wrapper for the index to be called by the Gradio interface.
    """

    def __init__(self):
        self.index: ExactMultiVecIndex = None
        self.dataset_path = None
        self.dataset_metadata = None
        self.image_embedding_generator = None
        self.text_embedding_generator = None

    def build_index(self, dataset_folder, text_weight, image_weight, text_metric, image_metric, index_type,
                    text_embedding_model, image_embedding_model):
        self.dataset_path = dataset_folder

        # load vectors from dataset_folder
        try:
            text_vectors, image_vectors = load_vectors_from_dataset_base_path(dataset_folder)
        except Exception as e:
            return f"Failed to load vectors from the dataset folder: {e}"

        print(
            f"Loaded 32-bit vectors. Text vectors shape: {text_vectors.shape}. Image vectors shape: {image_vectors.shape}.")

        # load metadata from dataset_folder
        try:
            self.dataset_metadata = pd.read_parquet(dataset_folder + "/metadata.parquet")
        except Exception as e:
            return "Failed to load metadata from the dataset folder. Please check the dataset path contains metadata.parquet: {e}"

        # initialise the embedding generators (used only for the search)
        self.text_embedding_generator = SentenceTransformerEmbeddingGenerator(model_name=text_embedding_model)
        self.image_embedding_generator = HFImageEmbeddingGenerator(model_name=image_embedding_model)

        # build index
        modalities = 2
        weights = [text_weight, image_weight]
        metrics = [text_metric, image_metric]
        dataset = [text_vectors, image_vectors]

        if index_type == "MultiVecHNSW":
            self.index = MultiVecHNSW(modalities, [text_vectors.shape[1], image_vectors.shape[1]], metrics, weights)
        elif index_type == "ExactMultiVecIndex":
            self.index = ExactMultiVecIndex(modalities, [text_vectors.shape[1], image_vectors.shape[1]], metrics,
                                            weights)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        self.index.add_entities(dataset)

        return f"Index successfully built! You can now search the index."

    def search(self, query_image, query_text, k, search_image_weight, search_text_weight):
        if self.index is None:
            return "Cannot search without building an index first. Please build the index.", None

        if query_image is None or query_text == "":
            return "Please provide an image and a text query.", None

        if k < 1:
            return "Please provide a positive integer value for k.", None

        # generate vectors for the query image and text
        query_text_vector = self.text_embedding_generator.generate_text_embeddings([query_text])[0]
        query_image_vector = self.image_embedding_generator.generate_image_embeddings([query_image])[0]
        query = [query_text_vector, query_image_vector]
        # search the index
        try:
            ids = self.index.search(query, k, query_weights=[search_text_weight, search_image_weight])
        except Exception as e:
            return f"Search failed with error: {e}", None

        # retrieve images and text with id from the dataset
        entity_results = []
        # load image and text
        images_path = self.dataset_path + "/images"
        for entity_id in ids:
            entity = (entity_id, load_image(entity_id, images_path), self.dataset_metadata["TEXT"][entity_id])
            entity_results.append(entity)

        return f"Search completed successfully. Returned {len(ids)} entities.", entity_results


# initialise the index wrapper that will be used through the demo
index_wrapper = IndexWrapper()

# main interface
with gr.Blocks(title="Multimodal Similarity Search Demo") as demo:
    gr.Markdown("# Multimodal Similarity Search Demonstration")

    # build index section
    with gr.Tab("Build Multimodal Index"):
        gr.Markdown("### Build the index from your dataset folder")
        with gr.Row():
            dataset_folder_input = gr.Textbox(label="Dataset Folder Path",
                                              placeholder="Enter the path to your dataset folder")
            dataset_folder_input.value = "/Users/yavuz/data/LAION-20000"
            index_type = gr.Dropdown(label="Index Type", choices=["ExactMultiVecIndex", "MultiVecHNSW"],
                                     value="ExactMultiVecIndex")

        with gr.Row():
            image_embedding_model = gr.Textbox(label="Image Embedding Model",
                                               value="google/vit-base-patch16-224-in21k")
            image_weight_slider = gr.Slider(0, 1, value=0.5, label="Image Weight")
            image_metric = gr.Dropdown(label="Image Metric", choices=["cosine", "Euclidean", "Manhattan"],
                                       value="cosine")

        with gr.Row():
            text_embedding_model = gr.Textbox(label="Text Embedding Model",
                                              value="BAAI/bge-small-en-v1.5")
            text_weight_slider = gr.Slider(0, 1, value=0.5, label="Text Weight")
            text_metric = gr.Dropdown(label="Text Metric", choices=["cosine", "Euclidean", "Manhattan"], value="cosine")

        build_button = gr.Button("Build Index")
        build_status = gr.Textbox(label="Status", placeholder="Build status will be shown here")

        build_button.click(fn=index_wrapper.build_index,
                           inputs=[dataset_folder_input, text_weight_slider, image_weight_slider, text_metric,
                                   image_metric, index_type, text_embedding_model, image_embedding_model],
                           outputs=build_status)

    # search index section
    with gr.Tab("Search Multimodal Index"):
        gr.Markdown("### Search the index")
        with gr.Row():
            query_image_input = gr.Image(label="Query Image", type="filepath")
            query_text_input = gr.Textbox(label="Query Text", placeholder="Enter text query")

        with gr.Row():
            search_image_weight_slider = gr.Slider(0, 1, value=image_weight_slider.value,
                                                   label="Image Weight for Search")
            search_text_weight_slider = gr.Slider(0, 1, value=text_weight_slider.value, label="Text Weight for Search")

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

        # k input and search button
        k_input = gr.Number(value=5, label="Number of Neighbours (k)", precision=0)
        search_button = gr.Button("Search")

        # output components
        search_status = gr.Textbox(label="Search Status")

        MAX_DISPLAYED_ENTITIES = 10
        result_containers = []
        for i in range(MAX_DISPLAYED_ENTITIES):
            with gr.Column(visible=False) as container:
                title = gr.Markdown(f"### Entity {i + 1}")
                with gr.Row():
                    img = gr.Image(label="Image", type="pil")
                    txt = gr.Textbox(label="Text", interactive=False)
            result_containers.append((container, title, img, txt))

    # function to update the containers with search results
    def display_search_results(query_image_path, query_text, k, search_image_weight, search_text_weight):
        status, entity_results = index_wrapper.search(query_image_path, query_text, k, search_image_weight,
                                                      search_text_weight)
        updates = [status]

        # if search fails, display only the status message
        if entity_results is None:
            for _ in range(MAX_DISPLAYED_ENTITIES):
                updates.extend([gr.update(visible=False), None, None, ""])
            return updates

        # update the rows of the UI with search results
        k = len(entity_results)
        for i in range(MAX_DISPLAYED_ENTITIES):
            if i < k:
                # update the container to show the image and text
                updates.extend([
                    gr.update(visible=True),
                    gr.Markdown(f"### Entity {entity_results[i][0]}"),  # display retrieved entity id
                    entity_results[i][1],  # image
                    entity_results[i][2]  # text
                ])
            else:
                # hide the row and clear image and text for unused rows
                updates.extend([gr.update(visible=False), None, None, ""])
        return updates


    # for each container we have 3 components: container update, image and text.
    outputs = [search_status] + [comp for container in result_containers for comp in container]
    search_button.click(fn=display_search_results,
                        inputs=[query_image_input, query_text_input, k_input, search_image_weight_slider,
                                search_text_weight_slider],
                        outputs=outputs)

demo.launch(share=False)
