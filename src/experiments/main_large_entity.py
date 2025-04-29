from multivec_index import MultiVecHNSW
import numpy as np
import random
import time
from tqdm import tqdm

from src.common.load_dataset import load_4_modality_dataset
from src.experiments.evaluation import IndexEvaluator, compute_exact_results, evaluate_index_construction, \
    evaluate_index_search, load_multivec_index_from_params, SEARCH_WEIGHTS_DIR, sanitise_path_string
from src.experiments.evaluation_params import Params, MultiVecHNSWConstructionParams, get_params_4_modality, get_params_3_modality, get_search_params
from datetime import datetime

from src.experiments.weights_experiments import evaluate_weighted_search_on_index

"""
4-modality experiments:
- number of distance computations - 0.25 weights and different metrics.
- cutoffs
- recall heatmap for 2 fixed weights at efSearch 100?
- 3D plot for weights and recall
"""


def get_index_and_query_vectors(all_vectors, num_indexed_entities, num_query_entities):
    """
    Given all vectors, returns the index and query vectors.
    Index vectors are the first num_indexed_entities vectors.
    Query vectors are the last num_query_entities vectors.
    """
    index_vectors = all_vectors[:num_indexed_entities]
    query_vectors = all_vectors[-num_query_entities:]
    return index_vectors, query_vectors


def main():
    # load dataset
    text_vectors_all, image_vectors_all, audio_vectors_all, video_vectors_all = load_4_modality_dataset()

    all_entities = [text_vectors_all, image_vectors_all, audio_vectors_all, video_vectors_all]
    map = {"text": 0, "image": 1, "audio": 2, "video": 3}

    # define modalities to index
    modalities = ["text", "image", "audio", "video"]

    # define subset for indexing and querying
    NUM_INDEXED_ENTITIES = 9_000
    NUM_QUERY_ENTITIES = 100

    # get index and query vectors
    index_and_query_vectors = [get_index_and_query_vectors(all_entities[map[modality]], NUM_INDEXED_ENTITIES, NUM_QUERY_ENTITIES) for modality in modalities]

    # define and build index that we will evaluate
    MODALITIES = 4
    DIMENSIONS = [all_entities[map[modality]].shape[1] for modality in modalities]
    DISTANCE_METRICS = ["cosine", "cosine", "cosine", "cosine"]
    WEIGHTS = [1 for _ in modalities]
    my_index = MultiVecHNSW(MODALITIES, dimensions=DIMENSIONS, distance_metrics=DISTANCE_METRICS, weights=WEIGHTS,
                         target_degree=32, max_degree=32, ef_construction=200, seed=1)
    my_index.set_ef_search(50)
    # search parameters
    k = 50

    # evaluate the index
    index_evaluator = IndexEvaluator(my_index)

    # evaluate inserting to the index
    index_entities = [index_and_query_vectors[map[modality]][0] for modality in modalities]
    insertion_time, memory_consumption = index_evaluator.evaluate_add_entities(index_entities)
    print(f"Insertion Time: {insertion_time:.3f} seconds.")
    print(f"Insertion Memory: {memory_consumption / 1024 / 1024} MiB.")

    # evaluate search performance
    queries = [index_and_query_vectors[map[modality]][1] for modality in modalities]
    search_times, recall_scores, memory_consumption = index_evaluator.evaluate_search(queries, k)

    print(
        f"Average search time: {np.mean(search_times)*1000:.3f}ms. Variance: {np.var(search_times)*1000*1000:.3f}ms^2. Min: {np.min(search_times)*1000:.3f}ms. Max: {np.max(search_times)*1000:.3f}ms.")
    print(
        f"Average recall: {np.mean(recall_scores):.3f}. Variance: {np.var(recall_scores):.3f}. Min: {np.min(recall_scores):.3f}. Max: {np.max(recall_scores):.3f}.")
    print(
        f"Average memory consumption: {np.mean(memory_consumption):.3f} bytes. Variance: {np.var(memory_consumption):.3f}. Min: {np.min(memory_consumption):.3f}. Max: {np.max(memory_consumption):.3f}.")



def evaluate_weights_experiments_4_modality(params, construction_params, index_size, ef_search_range, search_weight_tuples, normalised_dataset=False, num_query_entities=100, print_results=True):
    """
    Evaluate the effect of different search weights on the recall of the index.
    :param only_index_weights: if true, search based on only the weights the index was constructed with. If false, search with all weights from 0.0 to 1.0 (step 0.1)
    """
    search_params = get_search_params(params)

    k = params.k
    params.index_size = index_size
    search_params.k = k
    if normalised_dataset:
        multivec_index, index_file = load_multivec_index_from_params(params, construction_params, subdir_name="normalised/")
    else:
        multivec_index, index_file = load_multivec_index_from_params(params, construction_params)

    print(f"Loaded index with weights {multivec_index.weights} and metrics {multivec_index.distance_metrics}")

    # set query_ids to last NUM_QUERY_ENTITIES
    query_ids = random.sample(range(params.index_size, len(params.dataset[0])), num_query_entities)
    params.query_ids = query_ids
    print(f"Query_ids for index_size {index_size}:", query_ids)

    index_file_name = index_file.split('/')[-1].split('.')[0]
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]  # get up to millisecond

    save_folder = SEARCH_WEIGHTS_DIR
    if normalised_dataset:
        save_folder += "normalised/"
    save_folder += sanitise_path_string(f"{params.modalities}_{params.dimensions}_{params.metrics}_{params.weights}_{params.index_size}/") + \
                   f"{construction_params.target_degree}_{construction_params.max_degree}_{construction_params.ef_construction}_{construction_params.seed}/" + \
                   f"{index_file_name}/" + \
                   current_time + "/"

    for search_weights in tqdm(search_weight_tuples):
        search_params.weights = search_weights
        if print_results: print(f"Running weights experiments for index size, search weights: {index_size}, {search_params.weights}. (index weights: {params.weights})")

        exact_params = Params(params.modalities, params.dimensions, params.metrics, search_params.weights,
                              dataset=params.dataset, index_size=params.index_size, k=params.k,
                              query_ids=query_ids)
        if normalised_dataset:
            exact_results, exact_times = compute_exact_results(exact_params, cache=True, subdir_name="normalised/", print_results=print_results)
        else:
            exact_results, exact_times = compute_exact_results(exact_params, cache=True, print_results=print_results)

        # try values for ef_search=k', starting from ef_search=k
        for ef_search in ef_search_range:
            #for ef_search in range(300, 500, 10):
            search_params.ef_search = ef_search
            # search the index
            evaluate_weighted_search_on_index(multivec_index, exact_results, params, search_params, efs_folder=save_folder, print_results=print_results)


def generate_random_weights(n_samples):
    weights = []
    for i in range(n_samples):
        random_values = np.random.rand(3)
        random_values.sort()
        # compute weights as differences
        w1 = random_values[0]
        w2 = random_values[1] - random_values[0]
        w3 = random_values[2] - random_values[1]
        w4 = 1.0 - random_values[2]
        weights.append([w1, w2, w3, w4])

    # normalise weights
    weights = np.array(weights)
    weights = weights / np.sum(weights, axis=1, keepdims=True)

    return weights

def search_weights_exp_4_modalities():
    params = get_params_4_modality()
    construction_params = MultiVecHNSWConstructionParams(4, 8, 50, 400)
    index_weights =[[0.25, 0.25, 0.25, 0.25]]
    normalise = True
    index, index_path = evaluate_index_construction(params, construction_params, save_index=True, normalised=normalise)

    search_weight_tuples = generate_random_weights(1000)

    evaluate_weights_experiments_4_modality(params, construction_params, index_size=9000, ef_search_range=range(10,105,5), search_weight_tuples=search_weight_tuples, normalised_dataset=normalise, print_results=False)

def search_weights_exp_3_modalities():
    params = get_params_3_modality()
    construction_params = MultiVecHNSWConstructionParams(4, 8, 50, 400)
    normalise = True
    index, index_path = evaluate_index_construction(params, construction_params, save_index=True, normalised=normalise)

    search_weight_tuples = []
    for i in range(0, 11):
        for j in range(0, 11):
            if i + j <= 10:
                search_weight_tuples.append([i/10, j/10, (10-i-j)/10])

    evaluate_weights_experiments_4_modality(params, construction_params, index_size=9000, ef_search_range=range(10,105,5), search_weight_tuples=search_weight_tuples, normalised_dataset=normalise, print_results=False)


if __name__ == "__main__":
    #main()

    search_weights_exp_3_modalities()
