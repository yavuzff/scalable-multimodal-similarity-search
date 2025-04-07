#include "../include/MultiVecHNSW.hpp"
#include "../include/utils.hpp"
#include "../include/common.hpp"

#include <iostream>
#include <random>
#include <unordered_set>

using namespace std;
using entity_id_t = MultiVecHNSW::entity_id_t;

MultiVecHNSW::MultiVecHNSW(size_t numModalities,
                     vector<size_t> dims,
                     vector<string> distanceMetrics,
                     vector<float> weights,
                     float distributionScaleFactor,
                     size_t targetDegree,
                     size_t maxDegree,
                     size_t efConstruction,
                     size_t efSearch,
                     size_t seed): AbstractMultiVecIndex(numModalities, std::move(dims), std::move(distanceMetrics), std::move(weights)),
                                   distributionScaleFactor(distributionScaleFactor), targetDegree(targetDegree), maxDegree(maxDegree), efConstruction(efConstruction),
                                    efSearch(efSearch), seed(seed), maxLevel(0), maxDegreeLayer0(maxDegree*2), generator(seed) {
    validateParameters();

    // set distributionScaleFactor to good heuristic value if the default value (0) is being used
    if (distributionScaleFactor == 0.f) {
        this->distributionScaleFactor = 1.0f / log(targetDegree);
    }
    // initialise storage
    entityStorageByModality.resize(numModalities);

    // initialise stats
    if constexpr (TRACK_STATS) {
        num_compute_distance_calls = 0;
        num_lazy_distance_calls = 0;
        num_lazy_distance_cutoff = 0;
        num_vectors_skipped_due_to_cutoff = 0;
    }
}

void MultiVecHNSW::validateParameters() const {
    if (distributionScaleFactor < 0) {
        throw invalid_argument("Distribution scale factor must be positive");
    }
    // check that targetDegree is at least 1
    if (targetDegree < 1) {
        throw invalid_argument("Target degree must be at least 1");
    }
    if (maxDegree < targetDegree) {
        throw invalid_argument("Max degree must be at least the target degree");
    }
    if (efConstruction < 1) {
        throw invalid_argument("efConstruction must be at least 1");
    }
    if (efSearch < 1) {
        throw invalid_argument("efSearch must be at least 1");
    }
}

void MultiVecHNSW::addEntities(const vector<vector<float>>& entities) {
    addEntities(getSpanViewOfVectors(entities));
}

void MultiVecHNSW::addEntities(const vector<span<const float>>& entities) {
    const size_t numNewEntities = validateEntities(entities);

    std::cout << "Adding " << numNewEntities << " entities to MultiVecHNSW" << std::endl;
    //debug_printf("Adding %zu entities to MultiVecHNSW!\n", numNewEntities);
    addToEntityStorage(entities, numNewEntities);

    // allocate memory for the new nodes
    const entity_id_t finalNumEntities = numEntities + numNewEntities;
    assert (finalNumEntities > nodes.size());
    nodes.resize(finalNumEntities);

    // add each entity to the graph
    for (entity_id_t entityId = numEntities; entityId < finalNumEntities; ++entityId) {
        addEntityToGraph(entityId);
    }
    numEntities = finalNumEntities;

    if constexpr (TRACK_STATS) {
        std::cout << "num_compute_distance_calls: " << num_compute_distance_calls << " num_lazy_distance_calls: "
        << num_lazy_distance_calls << " num_lazy_distance_cutoff: " << num_lazy_distance_cutoff
        << " num_vectors_skipped_due_to_cutoff: " << num_vectors_skipped_due_to_cutoff <<  std::endl;
    }
}

void MultiVecHNSW::addToEntityStorageByModality(const vector<span<const float>>& entities, size_t numNewEntities) {
    // copy the input entities into the entityStorage
    for (size_t i = 0; i < numModalities; ++i) {
        const auto& modalityVectors = entities[i];
        entityStorageByModality[i].insert(entityStorageByModality[i].end(), modalityVectors.begin(), modalityVectors.end());

        // if the modality uses cosine distance, normalise the inserted vectors for this modality
        if (distanceMetrics[i] == DistanceMetric::Cosine) {
            cout << "Storing normalised vectors for modality " << i << " to efficiently compute cosine distance" << endl;
            for (size_t j = 0; j < numNewEntities; ++j) {
                l2NormalizeVector(span(entityStorageByModality[i]).subspan(j * dimensions[i], dimensions[i]));
            }
        }
    }
}

void MultiVecHNSW::addToEntityStorage(const vector<span<const float>>& entities, size_t numNewEntities) {
    // set the capacity of the entityStorage to the new size
    const size_t previousSize = entityStorage.size();
    const size_t numNewFloats = numNewEntities * totalDimensions;
    entityStorage.reserve(previousSize + numNewFloats);

    // copy the flattened input entities into entityStorage
    for (size_t i = 0; i < numNewEntities; ++i) {
        for (size_t modality = 0; modality < numModalities; ++modality) {
            size_t offset = i * dimensions[modality]; // offset to the start of the modality vectors
            entityStorage.insert(entityStorage.end(),
                                 entities[modality].begin() + offset,
                                 entities[modality].begin() + offset + dimensions[modality]);
        }
    }

    // normalise the inserted vectors for each modality that should be normalised
    size_t cumulativeDimensionSum = 0;
    for (size_t modality = 0 ; modality < numModalities; ++modality) {
        if (distanceMetrics[modality] == DistanceMetric::Cosine) {
            for (size_t i = 0; i < numNewEntities; ++i) {
                span<float> vectorToNormalise = span(entityStorage).subspan(previousSize + i * totalDimensions + cumulativeDimensionSum, dimensions[modality]);
                l2NormalizeVector(vectorToNormalise);
            }
        }
        cumulativeDimensionSum += dimensions[modality];
    }
}

span<const float> MultiVecHNSW::getEntityModalityFromEntityId(entity_id_t entityId, size_t modality) const {
    const vector<float>& modalityData = entityStorageByModality[modality];
    return span(modalityData.data() + entityId * dimensions[modality], dimensions[modality]);
}

std::span<const float> MultiVecHNSW::getEntityFromEntityId(entity_id_t entityId) const {
    return span(entityStorage.data() + entityId * totalDimensions, totalDimensions);
}

float MultiVecHNSW::computeDistance(const std::span<const float> entity1,  const std::span<const float> entity2, const std::vector<float>& weights) const {
    assert(entity1.size() == totalDimensions);
    assert(entity2.size() == totalDimensions);

    // incremented for tracking stats
    if constexpr (TRACK_STATS) num_compute_distance_calls++;

    float dist = 0.0f;
    size_t modalityStartIndex = 0;
    for (size_t modality = 0; modality < numModalities; ++modality) {
        if (weights[modality] != 0) {
            span<const float> vector1 = entity1.subspan(modalityStartIndex, dimensions[modality]);
            span<const float> vector2 = entity2.subspan(modalityStartIndex, dimensions[modality]);

            float modalityDistance;
            switch(distanceMetrics[modality]){
                case DistanceMetric::Euclidean:
                    modalityDistance = euclidean(vector1, vector2);
                break;
                case DistanceMetric::Manhattan:
                    modalityDistance = manhattan(vector1, vector2);
                break;
                case DistanceMetric::Cosine:
                    // vectors are already normalised
                    modalityDistance = cosine(vector1, vector2);
                break;
                default:
                    throw invalid_argument("Invalid distance metric. You should not be seeing this message.");
            }
            // check modalityDistance >= 0 with tolerance of 1e-6
            assert(modalityDistance >= -1e-6);

            // aggregate distance by summing modalityDistance*weight
            dist += weights[modality] * modalityDistance;
        }
        modalityStartIndex += dimensions[modality];
    }
    // check distance >= 0 with tolerance of 1e-6
    assert(dist >= -1e-6);
    return dist;
}

float MultiVecHNSW::computeDistanceLazy(const std::span<const float> entity1,  const std::span<const float> entity2, const std::vector<float>& weights, const float upperBound) const {
    assert(entity1.size() == totalDimensions);
    assert(entity2.size() == totalDimensions);

    if constexpr (TRACK_STATS) num_lazy_distance_calls++;

    float dist = 0.0f;
    size_t modalityStartIndex = 0;
    for (size_t modality = 0; modality < numModalities; ++modality) {
        if (weights[modality] != 0) {
            span<const float> vector1 = entity1.subspan(modalityStartIndex, dimensions[modality]);
            span<const float> vector2 = entity2.subspan(modalityStartIndex, dimensions[modality]);

            float modalityDistance;
            switch(distanceMetrics[modality]){
                case DistanceMetric::Euclidean:
                    modalityDistance = euclidean(vector1, vector2);
                break;
                case DistanceMetric::Manhattan:
                    modalityDistance = manhattan(vector1, vector2);
                break;
                case DistanceMetric::Cosine:
                    // vectors are already normalised
                    modalityDistance = cosine(vector1, vector2);
                break;
                default:
                    throw invalid_argument("Invalid distance metric. You should not be seeing this message.");
            }
            // check modalityDistance >= 0 with tolerance of 1e-6
            assert(modalityDistance >= -1e-6);

            // aggregate distance by summing modalityDistance*weight
            dist += weights[modality] * modalityDistance;

            // terminate early if the distance exceeds the upper bound
            if (dist >= upperBound) {
                if constexpr (TRACK_STATS) {
                    if (modality < numModalities - 1) {
                        num_lazy_distance_cutoff++; // track stat if actually avoided computation
                        num_vectors_skipped_due_to_cutoff += (numModalities-1) - modality;
                    }
                }
                return dist;
            }
        }
        modalityStartIndex += dimensions[modality];
    }
    // check distance >= 0 with tolerance of 1e-6
    assert(dist >= -1e-6);
    return dist;
}

vector<size_t> MultiVecHNSW::search(const vector<span<const float>>& query, size_t k, const vector<float>& queryWeights) {
    validateQuery(query, k);
    debug_printf("Searching MultiVecHNSW with query weights with k=%lu\n", k);
    // copy weights as we will normalise them
    auto normalisedQueryWeights = std::vector(queryWeights);
    validateAndNormaliseWeights(normalisedQueryWeights, numModalities);

    vector<entity_id_t> result = internalSearch(query, k, normalisedQueryWeights);
    return {result.begin(), result.end()};
}

vector<size_t> MultiVecHNSW::search(const vector<span<const float>>& query, size_t k) {
    validateQuery(query, k);
    debug_printf("Searching MultiVecHNSW without query weights with k=%lu\n", k);
    vector<entity_id_t> result = internalSearch(query, k, indexWeights);
    return {result.begin(), result.end()};
}

vector<size_t> MultiVecHNSW::search(const vector<vector<float>>& query, size_t k, const vector<float>& queryWeights) {
    return search(getSpanViewOfVectors(query), k, queryWeights);
}

vector<size_t> MultiVecHNSW::search(const vector<vector<float>>& query, size_t k) {
    return search(getSpanViewOfVectors(query), k);
}

void MultiVecHNSW::addEntityToGraph(const entity_id_t entityId) {
    debug_printf("Inserting entity %d \n", entityId);

    assert (entityId < nodes.size());
    const size_t targetLevel = generateRandomLevel();

    // allocate and initialise memory for the neighbours of the new node
    nodes[entityId].neighboursPerLayer.resize(targetLevel + 1);
    debug_printf("Resized neighboursPerLayer for the entity %d\n", entityId);
    // for layer 0, allocate vector of size maxDegreeLayer0
    nodes[entityId].neighboursPerLayer[0].reserve(maxDegreeLayer0);
    // for each layer above, allocate vector of size maxDegree
    for (size_t i = 1; i <= targetLevel; ++i) {
        nodes[entityId].neighboursPerLayer[i].reserve(maxDegree);
    }
    debug_printf("Reserved memory for each of the %zu layers\n", targetLevel + 1);

    if (entityId == 0) {
        // this is the first node in the graph
        debug_printf("Entity %d is the first node, setting as entry point\n", entityId);
        entryPoint = entityId;
        maxLevel = targetLevel;
        return;
    }

    // search graph from start node until the target layer
    priority_queue<pair<float, entity_id_t>> candidateNearestNeighbours;
    entity_id_t currentEntryPoint = entryPoint;
    debug_printf("Starting search for the nearest neighbors from entry point %d\n", currentEntryPoint);
    for (size_t layer = maxLevel; layer > targetLevel; --layer) {
        debug_printf("Searching in layer %zu\n", layer);
        candidateNearestNeighbours = searchLayer(entityId, {currentEntryPoint}, indexWeights, 1, layer);
        assert(candidateNearestNeighbours.size() == 1 && "Only one candidate should be returned from layer");
        currentEntryPoint = candidateNearestNeighbours.top().second; // get the only candidate, which is the closest element
    }

    // search layers at and below the target layer while inserting the new node
    const size_t firstLayer = min(maxLevel, targetLevel);
    debug_printf("Inserting into layers from %zu down to 0\n", firstLayer);
    for (size_t i = 0; i <= firstLayer; i++) {
        const size_t layer = firstLayer - i;
        debug_printf("Inserting to layer %zu from entry point %d\n", layer, currentEntryPoint);

        // currently have a single entry point to the layer below
        candidateNearestNeighbours = searchLayer(entityId, {currentEntryPoint}, indexWeights, efConstruction, layer);
        // selectNearestCandidates(candidateNearestNeighbours, targetDegree);
        selectDiversifiedCandidates(candidateNearestNeighbours, targetDegree, indexWeights);
        assert(candidateNearestNeighbours.top().first >= 0);
        assert(candidateNearestNeighbours.size() <= targetDegree);


        // identify the neighbours to connect the new node to
        vector<pair<float, entity_id_t>> neighboursToConnect;
        neighboursToConnect.reserve(candidateNearestNeighbours.size());
        while (!candidateNearestNeighbours.empty()) {
            neighboursToConnect.push_back(candidateNearestNeighbours.top());
            candidateNearestNeighbours.pop();
        }

        // update the entry point to be the closest neighbour, which is the last item since we appended from a max heap
        currentEntryPoint = neighboursToConnect.back().second;

        // add edges from the new node to the neighbours - note: we are adding the edges by decreasing distance
        for (pair<float, entity_id_t> neighbour : neighboursToConnect) {
            nodes[entityId].neighboursPerLayer[layer].push_back(neighbour.second);
            debug_printf("Connected node %d to neighbor %d at layer %zu\n", entityId, neighbour.second, layer);
        }
        addAndPruneEdgesForExistingNodes(entityId, neighboursToConnect, layer);
    }

    if (targetLevel > maxLevel) {
        debug_printf("Updating entry point to entity %d and max level to %zu\n", entityId, targetLevel);
        entryPoint = entityId;
        maxLevel = targetLevel;
    }
    debug_printf("Successfully added entity %d to the graph\n", entityId);
}

[[nodiscard]] size_t MultiVecHNSW::generateRandomLevel() const {
    // generate a random level using -ln(U(0,1)) * distributionScaleFactor
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    const double randomNumber = distribution(generator);
    const auto level = static_cast<size_t>(-log(randomNumber) * distributionScaleFactor);
    debug_printf("Generated level: %zu  from random number %f \n", level, randomNumber);
    return level;
}

void MultiVecHNSW::addAndPruneEdgesForExistingNodes(entity_id_t newEntityId, const vector<pair<float, entity_id_t>> &connectedNeighbours, size_t layer) {
    size_t _maxDegree;
    if (layer == 0) {
        _maxDegree = maxDegreeLayer0;
    } else {
        _maxDegree = maxDegree;
    }

    // iterate through neighbours which we connected to the new entity
    for (pair<float, entity_id_t> item: connectedNeighbours) {
        const entity_id_t currentEntity = item.second;
        if (nodes[currentEntity].neighboursPerLayer[layer].size() < _maxDegree) {
            // we have space so simply add the edge
            nodes[currentEntity].neighboursPerLayer[layer].push_back(newEntityId);
            debug_printf("Connected node %d to neighbor %d at layer %zu\n", currentEntity, newEntityId, layer);
        } else {
            // select the new set of edges to use
            assert (nodes[currentEntity].neighboursPerLayer[layer].size() == _maxDegree);
            debug_printf("Pruning edges for node %d at layer %zu\n", currentEntity, layer);

            bool simplyReplaceWorstNeighbour = false;
            if (simplyReplaceWorstNeighbour) {
                // replace worst neighbour
                size_t worstNeighbourIndex = _maxDegree;
                float maxDist = item.first;
                for (size_t i = 0; i < _maxDegree; i++) {
                    const float dist = computeDistance(
                                        getEntityFromEntityId(currentEntity),
                                        getEntityFromEntityId(nodes[currentEntity].neighboursPerLayer[layer][i]),
                                        indexWeights);
                    if (dist < maxDist) {
                        maxDist = dist;
                        worstNeighbourIndex = i;
                    }
                }
                if (worstNeighbourIndex != _maxDegree) {
                    nodes[currentEntity].neighboursPerLayer[layer][worstNeighbourIndex] = newEntityId;
                }
            } else {
                // use diversified candidates heuristic to select the new set of edges
                priority_queue<pair<float, entity_id_t>> candidateNeighbours;
                // first add entry for the new entity (item.first is distance(newEntityId, currentEntity))
                candidateNeighbours.emplace(item.first, newEntityId);

                // iterate through existing neighbours
                for (entity_id_t existingNeighbour: nodes[currentEntity].neighboursPerLayer[layer]) {
                    // compute distance for the corresponding edge - note: we would have computed this value previously when constructing edge
                    const float dist = computeDistance(
                                        getEntityFromEntityId(currentEntity),
                                        getEntityFromEntityId(existingNeighbour),
                                        indexWeights);
                    candidateNeighbours.emplace(dist, existingNeighbour);
                }

                selectDiversifiedCandidates(candidateNeighbours, _maxDegree, indexWeights);

                // update the neighbours for the current entity by resizing the vector and adding the new neighbours
                assert(nodes[currentEntity].neighboursPerLayer[layer].capacity() == _maxDegree);
                nodes[currentEntity].neighboursPerLayer[layer].clear();
                assert(nodes[currentEntity].neighboursPerLayer[layer].capacity() == _maxDegree);
                while (!candidateNeighbours.empty()) {
                    nodes[currentEntity].neighboursPerLayer[layer].push_back(candidateNeighbours.top().second);
                    candidateNeighbours.pop();
                }
            }
        }
    }
}


vector<entity_id_t> MultiVecHNSW::internalSearch(const vector<span<const float>>& userQuery, size_t k, const vector<float>& weights) const {
    if (numEntities == 0) {
        return {};
    }
    // copy and flatten the userQuery vectors into a single vector
    vector<float> flattenedQueryData;
    flattenedQueryData.reserve(totalDimensions);
    for (size_t i = 0; i < numModalities; ++i) {
        // copy vector into flattenedQueryData
        flattenedQueryData.insert(flattenedQueryData.end(), userQuery[i].begin(), userQuery[i].end());
        if (distanceMetrics[i] == DistanceMetric::Cosine) {
            // normalise query vector if this modality is using cosine distance
            assert(userQuery[i].size() == dimensions[i]);
            l2NormalizeVector(span(flattenedQueryData).subspan(flattenedQueryData.size() - dimensions[i], dimensions[i]));
        }
    }

    // create a span to the flattened query data, to use a unified searchLayer function
    const span<const float> query = flattenedQueryData;

    // start searching the graph
    priority_queue<pair<float, entity_id_t>> candidateNearestNeighbours;
    entity_id_t currentEntryPoint = entryPoint;
    for (size_t layer = maxLevel; layer > 0; --layer) {
        candidateNearestNeighbours = searchLayer(query, {currentEntryPoint}, weights, 1, layer);
        assert(candidateNearestNeighbours.size() == 1);
        currentEntryPoint = candidateNearestNeighbours.top().second;
    }
    // search layer 0
    candidateNearestNeighbours = searchLayer(query, {currentEntryPoint}, weights, max(efSearch, k), 0);

    // select the k nearest neighbours
    selectNearestCandidates(candidateNearestNeighbours, k);
    size_t numResults = candidateNearestNeighbours.size();
    vector<entity_id_t> result;
    result.resize(numResults);
    for (size_t i = 0; i < numResults; i++) {
        result[numResults - i - 1] = candidateNearestNeighbours.top().second;
        candidateNearestNeighbours.pop();
    }
    return result;
}

[[nodiscard]] priority_queue<pair<float, entity_id_t>> MultiVecHNSW::searchLayer(entity_id_t entityId, const vector<entity_id_t> &entryPoints, const vector<float>& weights, size_t ef, size_t layer) const {
    return searchLayer(getEntityFromEntityId(entityId), entryPoints, weights, ef, layer);
}

[[nodiscard]] priority_queue<pair<float, entity_id_t>> MultiVecHNSW::searchLayer(const std::span<const float> entity, const vector<entity_id_t> &entryPoints, const vector<float>& weights, size_t ef, size_t layer) const {
    assert(!entryPoints.empty());
    assert(layer <= maxLevel);
    // set of visited elements initialised to entryPoints
    unordered_set<entity_id_t> visited(entryPoints.begin(), entryPoints.end());
    // min priority queue of candidates (achieved by storing negative dist)
    priority_queue<pair<float, entity_id_t>> candidates;
    // fixed-size (ef) priority queue of nearest neighbours found so far, max heap
    priority_queue<pair<float, entity_id_t>> nearestNeighbours;

    // populate the priority queues
    for (entity_id_t entryPointId : entryPoints) {
        const std::span<const float> entryPoint = getEntityFromEntityId(entryPointId);
        const float dist = computeDistance(entryPoint, entity, weights);
        candidates.emplace(-dist, entryPointId);
        nearestNeighbours.emplace(dist, entryPointId);
    }

    while (!candidates.empty()) {
        auto [bestNegDist, bestCandidate] = candidates.top();
        float bestCandidateDist = -bestNegDist;
        candidates.pop();

        // terminate if the best candidate is worse than the worst neighbour
        auto [worstNeighbourDist, worstNeighbour] = nearestNeighbours.top();
        if (bestCandidateDist > worstNeighbourDist) {
            break;
        }

        // process the unvisited neighbours of the best candidate
        assert(nodes[bestCandidate].neighboursPerLayer.size() > layer); // candidate must be at least at this layer
        for (entity_id_t neighbourId : nodes[bestCandidate].neighboursPerLayer[layer]) {
            if (!visited.contains(neighbourId)) {
                visited.insert(neighbourId);
                const std::span<const float> neighbour = getEntityFromEntityId(neighbourId);

                // lazy compute distance and update the priority queues
                if (nearestNeighbours.size() < ef) {
                    // we will insert this neighbour, no matter the computed distance
                    const float newDist = computeDistance(neighbour, entity, weights);
                    candidates.emplace(-newDist, neighbourId);
                    nearestNeighbours.emplace(newDist, neighbourId);
                } else {
                    // we will lazy compute the distance and insert only if it is better than the worst neighbour
                    auto [_worstNeighbourDist, _worstNeighbour] = nearestNeighbours.top();
                    const float newDist = computeDistanceLazy(neighbour, entity, weights, _worstNeighbourDist);
                    if (newDist < _worstNeighbourDist) {
                        candidates.emplace(-newDist, neighbourId);
                        nearestNeighbours.emplace(newDist, neighbourId);
                        nearestNeighbours.pop();
                    }
                }
            }
        }
    }
    return nearestNeighbours;
}


void MultiVecHNSW::selectNearestCandidates(priority_queue<pair<float, entity_id_t>> &candidates, size_t resultSize) const {
    // precondition: candidates is a non-empty max heap
    assert(!candidates.empty());
    assert(candidates.top().first >= 0);

    // pop from the heap until needed
    while (candidates.size() > resultSize) {
        candidates.pop();
    }

    // // Note: selected elements are not in any particular order
    // vector<entity_id_t> selectedCandidates;
    // selectedCandidates.reserve(candidates.size());
    // for (const auto& pair : candidates) {
    //     selectedCandidates.push_back(pair.second);
    // }
}

// computes and finds the M closest entities to the targetEntityId from the candidates
priority_queue<pair<float, entity_id_t>> MultiVecHNSW::selectNearestCandidates(entity_id_t targetEntityId, const span<entity_id_t> candidates, size_t numSelected, const std::vector<float>& weights) const {
    // precondition: candidates is a non-empty array
    assert(!candidates.empty());
    assert(numSelected <= candidates.size());
    priority_queue<pair<float, entity_id_t>> maxHeap;
    for (const entity_id_t candidate : candidates) {
        assert(candidate != targetEntityId);

        // add candidate using lazy distance computation
        if (maxHeap.size() < numSelected) {
            const float dist = computeDistance(
            getEntityFromEntityId(targetEntityId),
            getEntityFromEntityId(candidate),
            weights);
            maxHeap.emplace(dist, candidate);
        } else {
            const float upperBound = maxHeap.top().first;
            const float dist = computeDistanceLazy(
                getEntityFromEntityId(targetEntityId),
                getEntityFromEntityId(candidate),
                weights,
                upperBound);
            if (dist < upperBound) {
                maxHeap.pop();
                maxHeap.emplace(dist, candidate);
            }
        }
    }
    assert(!maxHeap.empty() && maxHeap.size() <= numSelected);
    return maxHeap;
}


void MultiVecHNSW::selectDiversifiedCandidates(priority_queue<pair<float, entity_id_t>>& candidates, size_t targetSelectedNeighbours, const vector<float>& weights) const {
    // precondition: candidates is a non-empty max heap
    assert(!candidates.empty());
    assert(candidates.top().first >= 0);

    // we do not extend the list of candidates with neighbours of neighbours, which reduces insertion time
    // also, we will not force to return targetSelectedNeighbours, to avoid repeatedly pruning every time after we reach targetDegree for a node
    if (candidates.size() <= targetSelectedNeighbours) {
        return;
    }

    // form a min heap from the candidates
    priority_queue<pair<float, entity_id_t>> candidatesMinHeap;
    while (!candidates.empty()) {
        candidatesMinHeap.emplace(-candidates.top().first, candidates.top().second);
        candidates.pop();
    }

    vector<pair<float, entity_id_t>> selectedCandidates; // stores the diversified candidates
    while (!candidatesMinHeap.empty() && selectedCandidates.size() < targetSelectedNeighbours) {
        pair<float, entity_id_t> candidate = candidatesMinHeap.top();
        candidatesMinHeap.pop();
        const float distToCandidate = -candidate.first;

        // add current candidate only if (for all s in selected: dist(current, query) < dist(current, s)).
        bool addCandidate = true;
        for (const pair<float, entity_id_t>& selected : selectedCandidates) {
            const float distToSelected = computeDistanceLazy(
                getEntityFromEntityId(selected.second),
                getEntityFromEntityId(candidate.second),
                weights,
                distToCandidate);
            if (distToCandidate >= distToSelected) {
                addCandidate = false;
                break;
            }
        }
        if (addCandidate) {
            selectedCandidates.push_back(candidate);
        }
    }

    // update the original candidates heap
    for (const pair<float, entity_id_t>& selected : selectedCandidates) {
        candidates.emplace(-selected.first, selected.second);
    }
    assert(candidates.size() <= targetSelectedNeighbours);
    assert(candidates.top().first >= 0);
}

void MultiVecHNSW::printGraph() const {
    std::cout << "Printing graph layer by layer:\n";

    for (size_t layer = 0; layer <= maxLevel; ++layer) {

        std::cout << "Layer " << layer << ":\n";

        for (entity_id_t entityId = 0; entityId < nodes.size(); ++entityId) {
            // check if the current node has neighbors in this layer
            if (layer < nodes[entityId].neighboursPerLayer.size()) {

                // print the neighbors for the current node
                std::cout << "  Node " << entityId << " -> [";
                for (entity_id_t neighbor : nodes[entityId].neighboursPerLayer[layer]) {
                    std::cout << neighbor << " ";
                }
                std::cout << "]\n";
            }
        }

    }
}

void MultiVecHNSW::save(const string& path) const {
    cout << "Saving MultiVecHNSW to " << path << endl;
}

void MultiVecHNSW::load(const string& path) {
    cout << "Loading index from " << path << endl;
}

//implement getters and setters
float MultiVecHNSW::getDistributionScaleFactor() const {
    return distributionScaleFactor;
}

size_t MultiVecHNSW::getTargetDegree() const {
    return targetDegree;
}

size_t MultiVecHNSW::getMaxDegree() const {
    return maxDegree;
}

size_t MultiVecHNSW::getEfConstruction() const {
    return efConstruction;
}

size_t MultiVecHNSW::getEfSearch() const {
    return efSearch;
}

size_t MultiVecHNSW::getSeed() const {
    return seed;
}

void MultiVecHNSW::setEfSearch(size_t efSearch) {
    if (efSearch < 1) {
        throw invalid_argument("efSearch must be at least 1");
    }
    this->efSearch = efSearch;
}