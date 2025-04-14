import time
import numpy as np
import psutil
import os
from datetime import datetime

from multivec_index import ExactMultiVecIndex, MultiVecHNSW
from src.evaluation.evaluation_params import Params, MultiVecHNSWConstructionParams, MultiVecHNSWSearchParams
from src.common.load_dataset import load_dataset

EXPERIMENTS_DIR = "experiments/"
EXACT_RESULTS_DIR = EXPERIMENTS_DIR + "exact_results/"
CONSTRUCTION_DIR = EXPERIMENTS_DIR + "construction/"
SAVED_INDEX_DIR = EXPERIMENTS_DIR + "saved_index/"
SEARCH_DIR = EXPERIMENTS_DIR + "search/"
SEARCH_WEIGHTS_DIR = EXPERIMENTS_DIR + "search_weights_exps/"

RERANK_CONSTRUCTION_DIR = EXPERIMENTS_DIR + "rerank_construction/"
RERANK_SEARCH_DIR = EXPERIMENTS_DIR + "rerank_search/"

def compute_exact_results(p: Params, cache=True, recompute=True):
    """
    Return exact results for this set of inputs, caching if needed.
    """
    assert p.modalities == len(p.dataset)

    rounded_weights = [round(w, 5) for w in p.weights]
    save_folder = EXACT_RESULTS_DIR + sanitise_path_string(
        f"{p.modalities}_{p.dimensions}_{p.metrics}_{rounded_weights}_{p.index_size}_{p.k}/")

    if cache and os.path.exists(save_folder):
        # iterate over every folder in save_folder
        for folder in os.listdir(save_folder):
            # check folder is actually a folder
            if not os.path.isdir(save_folder + folder):
                continue
            query_path = save_folder + folder + "/query_ids.npy"
            if not os.path.exists(query_path):
                print(f"Warning: {query_path} not found, skipping")
            cached_query_ids = np.load(query_path)
            if np.array_equal(p.query_ids, cached_query_ids):
                print(f"Loading cached results from {save_folder + folder}")
                data = np.load(save_folder + folder + "/results.npz")
                return data["results"], data["search_times"]

    if not recompute:
        assert False, "No cached results found, and recompute is set to False. Exiting."

    # create directory where we will save our results
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]  # get up to millisecond
    save_folder += current_time + "/"
    os.makedirs(os.path.dirname(save_folder))

    search_times = []
    results = []

    exact_index = ExactMultiVecIndex(p.modalities, p.dimensions, p.metrics, p.weights)

    entities_to_insert = [modality[:p.index_size] for modality in p.dataset]
    exact_index.add_entities(entities_to_insert)
    print(f"Inserted {len(entities_to_insert[0])} entities to the exact index.")

    for query_id in p.query_ids:
        query = [modality[query_id] for modality in p.dataset]
        start_time = time.perf_counter()
        result = exact_index.search(query, p.k)
        end_time = time.perf_counter()
        search_times.append(end_time - start_time)
        results.append(result)

    # save query_ids, results and search_times at save_folder
    np.save(save_folder + "query_ids.npy", p.query_ids)
    np.savez(save_folder + "results.npz", results=results, search_times=search_times)

    print(f"Saved exact results.npz and query_ids.npy to {save_folder}")

    return np.array(results), np.array(search_times)

def load_multivec_index_from_params(params: Params, construction_params: MultiVecHNSWConstructionParams):
    index_string = sanitise_path_string(f"{params.modalities}_{params.dimensions}_{params.metrics}_{params.weights}_{params.index_size}/") + \
                   f"{construction_params.target_degree}_{construction_params.max_degree}_{construction_params.ef_construction}_{construction_params.seed}/"

    index_save_folder = SAVED_INDEX_DIR + index_string
    # load the latest file in the folder
    index_files = [f for f in os.listdir(index_save_folder) if f.startswith("index-")]
    assert len(index_files) > 0, f"No index files found in {index_save_folder}"
    index_file = index_files[-1]

    loaded_index = MultiVecHNSW(1, [1])
    loaded_index.load(index_save_folder+index_file)

    return loaded_index, index_file


def evaluate_index_construction(p: Params, specific_params: MultiVecHNSWConstructionParams, save_index=True):
    """
    Construct an index with the given parameters and save the time it took to construct it to a file.
    """
    index_string = sanitise_path_string(f"{p.modalities}_{p.dimensions}_{p.metrics}_{p.weights}_{p.index_size}/") + \
                   f"{specific_params.target_degree}_{specific_params.max_degree}_{specific_params.ef_construction}_{specific_params.seed}/"

    save_folder = CONSTRUCTION_DIR + index_string
    index_save_folder = SAVED_INDEX_DIR + index_string

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]  # get up to millisecond

    entities_to_insert = [modality[:p.index_size] for modality in p.dataset]

    start_time = time.perf_counter()
    multivec_hnsw = MultiVecHNSW(p.modalities, p.dimensions, p.metrics, weights=p.weights,
                           target_degree=specific_params.target_degree,
                           max_degree=specific_params.max_degree,
                           ef_construction=specific_params.ef_construction,
                           seed=specific_params.seed)
    multivec_hnsw.add_entities(entities_to_insert)
    total_time = time.perf_counter() - start_time

    os.makedirs(os.path.dirname(save_folder), exist_ok=True)
    save_file = save_folder + current_time + ".npz"
    np.savez(save_file, time=[total_time])

    if save_index:
        os.makedirs(os.path.dirname(index_save_folder), exist_ok=True)
        index_save_file = index_save_folder + "index-" + current_time + ".dat"
        multivec_hnsw.save(index_save_file)

    print(f"Constructed index in {total_time} seconds. Saved to {save_file}")

    return multivec_hnsw, save_file


def evaluate_index_search(index: MultiVecHNSW, index_path: str, exact_results, params: Params,
                          search_params: MultiVecHNSWSearchParams):
    """
    Search the index with the given parameters and save the ANN results and search times to a file.
    """
    assert params.k == search_params.k

    index_path_components = index_path.split('/')
    index_path_components[-1] = index_path_components[-1].replace(".npz", "")  # update the index path to be a folder
    rest_path = '/'.join(
        index_path_components[2:])  # do not include the starting part of the index path (exp/const/...)
    save_folder = SEARCH_DIR + rest_path + '/'

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]  # get up to millisecond
    save_folder += f"{search_params.k}_{search_params.ef_search}_" + current_time + "/"
    os.makedirs(os.path.dirname(save_folder))

    index.set_ef_search(ef_search=search_params.ef_search)

    search_times = []
    results = []
    recall_scores = []
    for i, query_id in enumerate(params.query_ids):
        query = [modality[query_id] for modality in params.dataset]
        start_time = time.perf_counter()
        result = index.search(query, params.k)
        end_time = time.perf_counter()
        if len(result) < params.k:
            print(f"WARNING: Search returned less than k={50} results. Returned {len(result)} results for {i}th query id: {query_id}. Padding with -1.")
            result = np.array(list(result) + [-1] * (params.k-len(result)))
        search_times.append(end_time - start_time)
        results.append(result)
        recall_scores.append(compute_recall(result, exact_results[i]))

    # save query_ids, results and search_times at save_folder
    np.save(save_folder + "query_ids.npy", params.query_ids)
    np.savez(save_folder + "results.npz", results=results, search_times=search_times,
             recall_scores=recall_scores, ef_search=search_params.ef_search)

    print(f"Search time, efSearch, recall: {sum(search_times) / len(search_times) * 1000:.3f}, {search_params.ef_search}, {sum(recall_scores) / len(recall_scores):.5f}")
    print(f"Saved efSearch={search_params.ef_search} results to {save_folder}")
    return results, search_times, recall_scores


def demonstrate_returning_less_than_k_results():
    # the HNSW index is not always guaranteed to return exactly k results, even when ef>k
    # this is because, under very unlikely construction circumstances,
    # (when targetDegree/efConstruction are relatively small compared to k)
    # for some query points, we might not actually traverse enough nodes to find k neighbours
    # the below demonstrates an example for degree=16, efConstruction=100, k=50

    from src.main import get_params, get_construction_params, get_search_params

    params = get_params()
    construction_params = get_construction_params()
    construction_params.target_degree = 16
    construction_params.max_degree = 16
    construction_params.ef_construction = 100
    construction_params.seed = 3

    search_params = get_search_params(params)
    params.index_size = 750_000

    params.k = 50
    search_params.k = 50

    query_ids = [1015111, 774477, 864130, 1004214, 786252, 916544, 1120053, 1153417, 943449, 1096350, 776621, 1000328, 981462, 1006975, 916798, 1100486, 893524, 797641, 932403, 1140023, 818774, 1038789, 851593, 1016775, 1079388, 775505, 1075601, 1109389, 845076, 912188, 1064972, 1102047, 857158, 1032182, 1162210, 894467, 1164696, 839803, 755904, 791900, 1091737, 1018669, 919532, 855596, 844430, 998130, 760181, 1040810, 959954, 997642, 904647, 1076740, 954817, 1028458, 877197, 1042576, 825826, 1048271, 1018341, 984439, 800293, 769220, 839112, 1001845, 1117548, 751826, 976962, 949483, 1146829, 872892, 753442, 760480, 1074141, 978458, 1101354, 913599, 759107, 1080280, 798161, 1099793, 914928, 1000477, 1098178, 885440, 1031465, 928890, 1162229, 1129433, 773231, 996061, 803415, 980995, 933579, 860739, 954478, 885272, 1122906, 1116602, 775520, 899094, 1028572, 999544, 885068, 1154960, 904169, 871051, 993051, 768466, 841538, 976134, 956792, 1056349, 1169294, 1023786, 841185, 879950, 869443, 779958, 1035824, 774253, 1030636, 1013137, 901879, 766192, 1158028, 757007, 885811, 893865, 803208, 1165891, 857216, 1005582, 863014, 1074818, 883788, 1147971, 1132451, 1061258, 1117437, 1135787, 783360, 1002123, 1099376, 1058999, 1147431, 954239, 1086491, 785688, 863986, 1054325, 1018319, 825232, 778196, 1138780, 896799, 995359, 903894, 827208, 901652, 1095201, 1146496, 776984, 1040132, 1115967, 1038079, 1064502, 975967, 827393, 1046998, 846410, 1120530, 931855, 1146390, 873028, 908885, 1004484, 994653, 1157640, 1089383, 1086451, 929283, 878269, 1064104, 883410, 925647, 940777, 957564, 1111185, 978216, 867278, 1160598, 904091, 853752, 1031489, 962780, 937307, 1089321, 928637, 1139479, 961284, 843632, 846221, 1148718, 792598, 877683, 764917, 1125253, 1156041, 1050813, 776074, 958362, 1057127, 788687, 1068558, 1025783, 879698, 756798, 1045785, 1020696, 1067410, 892375, 1091748, 883085, 1020539, 999380, 970106, 1117346, 855637, 913559, 761119, 909014, 1102838, 838725, 862697, 1064264, 833133, 931558, 1151663, 1091987, 853153, 834497, 797633, 1043210, 971799, 798064, 1070642, 991813, 900675, 855130, 1123425, 817879, 814296, 988521, 1032743, 1165672, 1014232, 1021462, 1090352, 912533, 780603, 1027825, 1030129, 998984, 883575, 950320, 1024669, 963686, 1020663, 850942, 826935, 822449, 841459, 912848, 822981, 990933, 884761, 783627, 960277, 970998, 767597, 1169282, 1146712, 868565, 1124643, 979373, 941942, 919871, 1145449, 871868, 1103601, 1161677, 989959, 1057374, 1117344, 813122, 820034, 832474, 1045085, 1112436, 1142804, 758431, 1131923, 811671, 769193, 1123146, 880763, 767333, 775310, 887197, 879606, 981514, 992538, 796763, 838636, 1061343, 831198, 1093361, 1022854, 801819, 868233, 1134057, 1152937, 957971, 888642, 923402, 1164794, 999583, 990876, 902005, 797279, 950723, 888375, 994352, 1123083, 948469, 761177, 939683, 1127538, 888120, 1044092, 1113325, 944230, 1028841, 1027875, 982658, 1163041, 1031976, 1058329, 1167275, 1164655, 1138584, 1069817, 957293, 777092, 860529, 925740, 892954, 886876, 859135, 1056631, 761831, 989266, 895212, 782935, 936862, 979702, 835345, 782836, 1130249, 811593, 1098787, 880862, 1010430, 841200, 1091890, 825887, 977022, 1097206, 885077, 1091488, 930953, 1137108, 898770, 828000, 791358, 860929, 1010938, 876196, 959021, 1067962, 825112, 957701, 800205, 852124, 1131409, 1007590, 1123409, 848794, 966149, 1147249, 783499, 876644, 1146878, 1100930, 1126830, 886560, 1157933, 942178, 1100122, 932421, 1008049, 945473, 756187, 894942, 798287, 785213, 959460, 1135147, 856130, 1152019, 1140335, 1133149, 964767, 1103807, 1025196, 796254, 780805, 975471, 950528, 958291, 867097, 947659, 1016649, 995897, 1083630, 807077, 868296, 761574, 879837, 955611, 1076765, 929255, 1026365, 1096070, 1099823, 1105126, 980645, 1021902, 1075222, 1012147, 1124851, 1032776, 1104235, 839339, 1149105, 815055, 1102734, 977297, 959737, 780281, 1168885, 1052961, 1063781, 944059, 775901, 1132230, 754955, 1093852, 764380, 922629, 758283, 866867, 781114, 1087138, 787593, 957165, 896224, 853279, 1151164, 932295, 1043566, 989758, 1098232, 1040129, 1154216, 1042842, 971240, 964869, 954614, 832706, 866885, 992967, 1071343, 1049525, 1161954, 986568, 914839, 994700, 954258, 1033804, 898755, 1018160, 1092730, 1025906, 1008612, 812318, 1130848, 1030263, 818103, 1139122, 896573, 816985, 755714, 1101448, 956379, 1156740, 1158511, 1029593, 1145285, 886140, 972092, 944135, 935004, 1035640, 918241, 804126, 838664, 1052642, 750641, 927185, 823263, 877493, 1138245, 868558, 914894, 969690, 1045792, 809480, 919428, 790320, 992316, 833378, 798614, 883248, 956688, 794513, 764564, 1140938, 1062899, 1058579, 941685, 887696, 1139678, 971975, 1115671, 913186, 1027246, 867656, 810213, 881406, 1058764, 1113717, 850224, 1116422, 1022686, 961750, 869153, 1060175, 1105829, 1029459, 886780, 983887, 876339, 785711, 1102139, 979932, 817456, 1087614, 1145919, 764558, 813767, 1061009, 776362, 1006426, 1044291, 790599, 1046974, 870838, 1068221, 781447, 1169270, 1159298, 770418, 954541, 966738, 1115896, 1056617, 1151165, 1041823, 982913, 852998, 1105563, 763163, 1142515, 1152475, 850585, 888825, 819537, 1150921, 810612, 1030789, 963116, 1019891, 768154, 888244, 1048198, 805086, 869215, 1132111, 863547, 1092136, 1016739, 935180, 894306, 1017839, 783114, 831872, 797679, 879244, 914236, 1142734, 1045190, 817829, 952412, 770257, 785759, 837180, 795491, 861293, 870716, 1103851, 1064035, 877900, 942146, 1134970, 874551, 791480, 1025763, 1100453, 1069449, 978507, 854974, 805852, 900599, 900020, 1032616, 855149, 802420, 894268, 768218, 1045996, 1018815, 932108, 931128, 966518, 919280, 995634, 883423, 1048313, 851504, 1062680, 1115067, 928009, 1120482, 818941, 1117833, 1144389, 1040045, 846812, 1059380, 865249, 1093154, 1138246, 1089414, 939566, 750015, 967804, 1057769, 876727, 783674, 1055305, 1094121, 788440, 1085276, 902292, 1016916, 769804, 861223, 1018427, 811732, 879781, 775865, 843273, 1158126, 901989, 873855, 822419, 879913, 957974, 978845, 1151891, 976444, 801191, 1129842, 1150782, 1166673, 1105992, 1077497, 1160663, 1074280, 1053567, 978552, 1119497, 800684, 1014335, 1063085, 1008798, 1125414, 867484, 1130430, 1079836, 873449, 1059390, 1001898, 761412, 928913, 859149, 823019, 1088948, 882829, 925580, 1057719, 853579, 814777, 813486, 1134554, 971632, 1091230, 1078517, 1149828, 920800, 894726, 1089655, 777573, 1024857, 821898, 760106, 816554, 832379, 1128798, 1092410, 978943, 969124, 781340, 1098406, 848958, 1006300, 1120033, 952697, 1057927, 785668, 1046860, 1071559, 1081688, 1109522, 755534, 1075523, 871188, 1167673, 993014, 1036393, 932713, 783920, 772213, 980214, 1085842, 987160, 801504, 1047781, 942660, 1126385, 826816, 1033738, 1032904, 779898, 972741, 1032054, 893455, 930410, 1145209, 937909, 760527, 1152537, 876268, 1110526, 994799, 953135, 1167081, 1019400, 860695, 811313, 997482, 975991, 784458, 795313, 1005485, 1009681, 1002061, 795577, 751933, 845303, 978973, 973190, 837032, 755415, 916533, 1144932, 1082530, 768542, 1121106, 827224, 750622, 1015800, 976740, 790774, 1034957, 954229, 798715, 936423, 1078607, 854458, 1016463, 918943, 1096848, 906014, 1026712, 960549, 1110572, 762508, 1109551, 1131957, 1020567, 1018757, 1026327, 1041241, 775934, 1055352, 905793, 1066834, 1021236, 761823, 1160555, 927994, 1032807, 799249, 983068, 1010543, 963545, 1147087, 1097797, 905504, 1037606, 940133, 949850, 1162874, 1163266, 836676, 939639, 1091026, 798406, 1140660, 1050045, 886652, 887675, 977780, 1158961, 775760, 1167193, 1136845, 804853, 1060947, 766095, 922712, 1072640, 962325, 1082664, 1167715, 1118457, 998916, 938609, 1005079, 750583, 1129907, 1031690, 874938, 893310, 1113434, 1064510, 905338, 902424, 1012107, 768571, 909834, 809981, 1032307, 876404, 933798, 828578, 1168170, 1136293, 770795, 1064735, 1109671, 1052021, 1015955, 922655, 1115842, 998330, 1085910, 1010028, 888514, 1038004, 797907, 1111483, 908993, 769231, 1144079, 1017625, 846941, 819214, 880527, 767007, 905579, 872007, 1161369, 1016335, 910951, 852849, 1069188, 1024934, 949536, 889244, 799811, 1030978, 1011654, 1050132, 993901, 900082, 772208, 768440, 805298, 829684, 911104, 970189, 1035963, 1128533, 994133, 879856, 965817, 1025914, 828560, 1003300, 911384, 848886, 974658, 795079, 1041606, 961488, 1012051, 763553, 772933, 814592, 860836, 829619, 1008181, 1041439, 888728, 949415, 772345, 906521, 983175, 1103892, 1038591, 1074457, 1055319, 814831, 994208, 919276, 917777, 1125612]
    params.query_ids = query_ids

    exact_results, exact_times = compute_exact_results(params, cache=True)  # will cache these when possible

    index = MultiVecHNSW(1,[1])
    index.load("experiments/saved_index/2_:384,768:_:cosine,cosine:_:0.5,0.5:_750000/anomaly-16_16_100_3/index-2025-04-10_07-10-49-806.dat")

    index_path = "experiments/construction/2_:384,768:_:cosine,cosine:_:0.5,0.5:_750000/anomaly-16_16_100_3/2025-04-10_07-10-49-806.npz"

    search_params.ef_search = 400
    results, search_times, recall_scores = evaluate_index_search(index, index_path,
                                                                     exact_results, params,
                                                                     search_params)

def evaluate_single_modality():
    """
    Evaluate the MultiVecHSNW index construction and search for a single modality.
    """
    import time

    text_vectors_all, image_vectors_all = load_dataset()

    MODALITIES = 1
    DIMENSIONS = [text_vectors_all.shape[1]]
    DISTANCE_METRICS = ["cosine"]
    WEIGHTS = [1]

    entities_to_insert = [text_vectors_all[:100_000]]

    start_time = time.perf_counter()
    multivec_hnsw = MultiVecHNSW(MODALITIES, DIMENSIONS, DISTANCE_METRICS, weights=WEIGHTS,
                           target_degree=16,
                           max_degree=16,
                           ef_construction=200,
                           seed=10)
    multivec_hnsw.add_entities(entities_to_insert)
    total_time = time.perf_counter() - start_time

    print(f"Index construction time: {total_time}")


def evaluate_hnsw_rerank_construction(p: Params, specific_params: MultiVecHNSWConstructionParams):
    """
    Evaluate the construction and search of a HNSW index with reranking.
    """
    print("Starting rerank construction at ", datetime.now(), " for ", p.dimensions, " and ", p.metrics)

    save_folder = RERANK_CONSTRUCTION_DIR + \
                  sanitise_path_string(f"{p.modalities}_{p.dimensions}_{p.metrics}_{p.weights}_{p.index_size}/") + \
                  f"{specific_params.target_degree}_{specific_params.max_degree}_{specific_params.ef_construction}_{specific_params.seed}/"

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]  # get up to millisecond

    indexes = []
    construction_times = []
    for i in range(p.modalities):
        vectors_to_insert = [p.dataset[i][:p.index_size]]
        start_time = time.perf_counter()
        index = MultiVecHNSW(1, [p.dimensions[i]], [p.metrics[i]], weights=[1],
                               target_degree=specific_params.target_degree,
                               max_degree=specific_params.max_degree,
                               ef_construction=specific_params.ef_construction,
                               seed=specific_params.seed)
        index.add_entities(vectors_to_insert)
        total_time = time.perf_counter() - start_time
        construction_times.append(total_time)
        indexes.append(index)

    print(f"Constructed indexes in: {construction_times} with total time {sum(construction_times)}")
    print(f"Index params were {specific_params.target_degree}, {specific_params.max_degree}, {specific_params.ef_construction}, {specific_params.seed}")

    # save construction times
    os.makedirs(os.path.dirname(save_folder), exist_ok=True)
    save_file = save_folder + current_time + ".npz"
    np.savez(save_file, time=construction_times)
    print(f"Saved construction times to {save_file}")

    return indexes, save_file


def evaluate_hnsw_rerank_search(indexes, index_path: str, exact_results, params: Params, search_params: MultiVecHNSWSearchParams):
    """
    Evaluate the search performance of a HNSW index with reranking. We use ef search as the k for each search.
    """
    index_path_components = index_path.split('/')
    index_path_components[-1] = index_path_components[-1].replace(".npz", "")  # update the index path to be a folder
    rest_path = '/'.join(
        index_path_components[2:])  # do not include the starting part of the index path (exp/construction/...)
    save_folder = RERANK_SEARCH_DIR + rest_path + '/'

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]  # get up to millisecond
    save_folder += f"{search_params.k}_{search_params.ef_search}_" + current_time + "/"
    os.makedirs(os.path.dirname(save_folder))

    for index in indexes:
        index.set_ef_search(search_params.ef_search)

    # note that we do not keep track of the results returned, just the recall
    search_times = []
    recall_scores = []
    for i, query_id in enumerate(params.query_ids):
        start_time = time.perf_counter()
        ids = []
        for modality in range(0, len(indexes)):
            result = indexes[modality].search([params.dataset[modality][query_id]], search_params.ef_search)
            ids.append(result)
        # flatten the ids into a set containing unique elements
        result = set([item for sublist in ids for item in sublist])
        end_time = time.perf_counter()

        search_times.append(end_time - start_time)
        recall_scores.append(compute_recall(result, exact_results[i]))

    # save query_ids, recall_scores and search_times at save_folder
    np.save(save_folder + "query_ids.npy", params.query_ids)
    np.savez(save_folder + "results.npz", search_times=search_times,
             recall_scores=recall_scores, ef_search=search_params.ef_search)

    print(f"Rerank: Search time, efSearch=k', recall: {sum(search_times) / len(search_times) * 1000:.3f}, {search_params.ef_search}, {sum(recall_scores) / len(recall_scores):.5f}")


def sanitise_path_string(path):
    """
    Replace invalid characters in a path string.
    """
    return path.replace(" ", "").replace("'", "").replace("[", ":").replace("]", ":")


class IndexEvaluator:
    """
    Class for evaluating the performance of an index, by computing exact index results and comparing them to evaluated index results.
    """
    def __init__(self, index: MultiVecHNSW):
        self.index = index

        # create an exact index for recall calculation
        self.exact_index = ExactMultiVecIndex(
            num_modalities=index.num_modalities,
            dimensions=index.dimensions,
            distance_metrics=index.distance_metrics,
            weights=index.weights
        )

        self.process = psutil.Process(os.getpid())

    def evaluate_add_entities(self, entities: list[np.ndarray]) -> tuple[float, int]:
        """
        Evaluates the time it takes to add entities to the index.
        :arg entities: A list of numpy arrays, each containing the entity vectors for a modality.
        :returns: The total time it took to add the entities to the index and the max memory usage in bytes.
        """
        mem_before = self.process.memory_info().rss
        start_time = time.perf_counter()
        self.index.add_entities(entities)
        total_time = time.perf_counter() - start_time
        mem_after = self.process.memory_info().rss
        mem_usage = mem_after - mem_before

        # also add it to exact index
        mem_before = self.process.memory_info().rss
        exact_index_start_time = time.perf_counter()
        self.exact_index.add_entities(entities)
        exact_index_total_time = time.perf_counter() - exact_index_start_time
        mem_after = self.process.memory_info().rss
        print(f"Exact index insertion time: {exact_index_total_time:.3f} seconds.")
        print(f"Exact index memory consumption: {(mem_after - mem_before) / 1024 / 1024} MiB.")

        return total_time, mem_usage

    def evaluate_search(self, queries: list[np.ndarray], k: int):
        """
        Evaluates the search performance of the index.
        :arg queries: A list of numpy arrays, each containing the query vectors for a modality.
        :arg k: The number of nearest-neighbors to search for.
        :returns: A tuple containing the search times and recall scores.
        """
        num_queries = len(queries[0])
        search_times = []
        recall_scores = []
        memory_consumptions = []
        for i in range(num_queries):
            query = [modality[i] for modality in queries]

            mem_before = self.process.memory_info().rss
            start_time = time.perf_counter()
            results = self.index.search(query, k)
            end_time = time.perf_counter()
            mem_after = self.process.memory_info().rss

            search_times.append(end_time - start_time)
            memory_consumptions.append(mem_after - mem_before)

            # calculate recall
            exact_results = self.exact_index.search(query, k)
            recall_scores.append(compute_recall(results, exact_results))

        return search_times, recall_scores, memory_consumptions


def compute_recall(results, exact_results):
    """
    Compute the recall of the results given the exact results.
    :arg results: The results obtained from the index.
    :arg exact_results: The exact results for the query.
    :returns: The recall of the results.
    """
    return len(set(results).intersection(set(exact_results))) / len(exact_results)
