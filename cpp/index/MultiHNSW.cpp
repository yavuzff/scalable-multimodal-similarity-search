#include "../include/MultiHNSW.hpp"
#include "../include/utils.hpp"
#include "../include/common.hpp"

#include <iostream>
#include <random>
#include <unordered_set>

using namespace std;
using entity_id_t = MultiHNSW::entity_id_t;

MultiHNSW::MultiHNSW(size_t numModalities,
                     vector<size_t> dims,
                     vector<string> distanceMetrics,
                     vector<float> weights,
                     float distributionScaleFactor,
                     size_t targetDegree,
                     size_t maxDegree,
                     size_t efConstruction,
                     size_t efSearch,
                     size_t seed): AbstractMultiIndex(numModalities, std::move(dims), std::move(distanceMetrics), std::move(weights)),
                                   distributionScaleFactor(distributionScaleFactor), targetDegree(targetDegree), maxDegree(maxDegree), efConstruction(efConstruction),
                                    efSearch(efSearch), seed(seed), maxLevel(0), maxDegreeLayer0(maxDegree*2), generator(seed) {
    validateParameters();
    // initialise storage
    entityStorage.resize(numModalities);
}

void MultiHNSW::validateParameters() const {
    if (distributionScaleFactor <= 0) {
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

void MultiHNSW::addEntities(const vector<vector<float>>& entities) {
    addEntities(getSpanViewOfVectors(entities));
}

void MultiHNSW::addEntities(const vector<span<const float>>& entities) {
    const size_t numNewEntities = validateEntities(entities);

    debug_printf("Adding %zu entities to MultiHNSW!\n", numNewEntities);
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
}

void MultiHNSW::addToEntityStorage(const vector<span<const float>>& entities, size_t numNewEntities) {
    // copy the input entities into the entityStorage
    for (size_t i = 0; i < numModalities; ++i) {
        const auto& modalityVectors = entities[i];
        entityStorage[i].insert(entityStorage[i].end(), modalityVectors.begin(), modalityVectors.end());

        // if the modality uses cosine distance, normalise the inserted vectors for this modality
        if (distanceMetrics[i] == DistanceMetric::Cosine) {
            cout << "Storing normalised vectors for modality " << i << " to efficiently compute cosine distance" << endl;
            for (size_t j = 0; j < numNewEntities; ++j) {
                l2NormalizeVector(span(entityStorage[i]).subspan(j * dimensions[i], dimensions[i]));
            }
        }
    }
}

span<const float> MultiHNSW::getEntityModality(entity_id_t entityId, entity_id_t modality) const {
    const vector<float>& modalityData = entityStorage[modality];
    return span(modalityData.data() + entityId * dimensions[modality], dimensions[modality]);
}

float MultiHNSW::computeDistanceBetweenEntities(entity_id_t entityId1, entity_id_t entityId2, const vector<float>& weights) const{
    float dist = 0.0f;
    for (size_t modality = 0; modality < numModalities; ++modality) {
        if (weights[modality] == 0) {
            continue;
        }
        span<const float> vector1 = getEntityModality(entityId1, modality);
        span<const float> vector2 = getEntityModality(entityId2, modality);

        float modalityDistance;
        switch(distanceMetrics[modality]){
            case DistanceMetric::Euclidean:
                modalityDistance = computeEuclideanDistance(vector1, vector2);
                break;
            case DistanceMetric::Manhattan:
                modalityDistance = computeManhattanDistance(vector1, vector2);
                break;
            case DistanceMetric::Cosine:
                // vectors are already normalised
                modalityDistance = computeCosineDistance(vector1, vector2, true);
                break;
            default:
                throw invalid_argument("Invalid distance metric. You should not be seeing this message.");
        }
        assert(modalityDistance >= 0);
        // aggregate distance by summing modalityDistance*weight
        dist += weights[modality] * modalityDistance;
    }
    assert (dist >= 0);
    return dist;
}

float MultiHNSW::computeDistanceToQuery(entity_id_t entityId, const vector<span<const float>>& query, const vector<float>& weights) const {
    float dist = 0.0f;
    for (size_t modality = 0; modality < numModalities; ++modality) {
        if (weights[modality] == 0) {
            continue;
        }
        span<const float> vector1 = getEntityModality(entityId, modality);
        span<const float> vector2 = query[modality];

        float modalityDistance;
        switch(distanceMetrics[modality]){
            case DistanceMetric::Euclidean:
                modalityDistance = computeEuclideanDistance(vector1, vector2);
            break;
            case DistanceMetric::Manhattan:
                modalityDistance = computeManhattanDistance(vector1, vector2);
            break;
            case DistanceMetric::Cosine:
                // vectors are already normalised
                modalityDistance = computeCosineDistance(vector1, vector2, true);
            break;
            default:
                throw invalid_argument("Invalid distance metric. You should not be seeing this message.");
        }
        assert(modalityDistance >= 0);

        // aggregate distance by summing modalityDistance*weight
        dist += weights[modality] * modalityDistance;
    }
    assert(dist >= 0);
    return dist;
}


vector<size_t> MultiHNSW::search(const vector<span<const float>>& query, size_t k, const vector<float>& queryWeights) {
    validateQuery(query, k);
    debug_printf("Searching MultiHNSW with query weights with k=%lu\n", k);
    // copy weights as we will normalise them
    auto normalisedQueryWeights = std::vector(queryWeights);
    validateAndNormaliseWeights(normalisedQueryWeights, numModalities);

    vector<entity_id_t> result = internalSearch(query, k, normalisedQueryWeights);
    return {result.begin(), result.end()};
}

vector<size_t> MultiHNSW::search(const vector<span<const float>>& query, size_t k) {
    validateQuery(query, k);
    debug_printf("Searching MultiHNSW without query weights with k=%lu\n", k);
    vector<entity_id_t> result = internalSearch(query, k, indexWeights);
    return {result.begin(), result.end()};
}

vector<size_t> MultiHNSW::search(const vector<vector<float>>& query, size_t k, const vector<float>& queryWeights) {
    return search(getSpanViewOfVectors(query), k, queryWeights);
}

vector<size_t> MultiHNSW::search(const vector<vector<float>>& query, size_t k) {
    return search(getSpanViewOfVectors(query), k);
}

void MultiHNSW::addEntityToGraph(entity_id_t entityId) {
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
        // this is the first node
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
        selectNearestCandidates(candidateNearestNeighbours, targetDegree);

        // identify the neighbours to connect the new node to
        vector<pair<float, entity_id_t>> neighboursToConnect;
        neighboursToConnect.reserve(candidateNearestNeighbours.size());
        while (!candidateNearestNeighbours.empty()) {
            neighboursToConnect.push_back(candidateNearestNeighbours.top());
            candidateNearestNeighbours.pop();
        }

        // update the entry point to be the closest neighbour, which is the last item since we appended from a max heap
        currentEntryPoint = neighboursToConnect.back().second;

        // add edges from the new node to the neighbours
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

[[nodiscard]] size_t MultiHNSW::generateRandomLevel() const {
    // generate a random level using -ln(U(0,1)) * distributionScaleFactor
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    const double randomNumber = distribution(generator);
    const auto level = static_cast<size_t>(-log(randomNumber) * distributionScaleFactor);
    debug_printf("Generated level: %zu  from random number %f \n", level, randomNumber);
    return level;
}

void MultiHNSW::addAndPruneEdgesForExistingNodes(entity_id_t newEntityId, vector<pair<float, entity_id_t>> &connectedNeighbours, size_t layer) {
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

            // replace worst neighbour
            size_t worstNeighbourIndex = _maxDegree;
            float maxDist = item.first;
            for (size_t i = 0; i < _maxDegree; i++) {
                const float dist = computeDistanceBetweenEntities(currentEntity, nodes[currentEntity].neighboursPerLayer[layer][i], indexWeights);
                if (dist < maxDist) {
                    maxDist = dist;
                    worstNeighbourIndex = i;
                }
            }
            if (worstNeighbourIndex != _maxDegree) {
                nodes[currentEntity].neighboursPerLayer[layer][worstNeighbourIndex] = newEntityId;
            }

            /*
            priority_queue<pair<float, entity_id_t>> neighboursToConsider;
            //first add entry for the newEntity: Assuming distance is symmetric?
            neighboursToConsider.emplace(item.first, newEntityId);
            // iterate through existing neighbours
            for (entity_id_t existingNeighbour: nodes[currentEntity].neighboursPerLayer[layer]) {
                const float dist = computeDistanceBetweenEntities(currentEntity, existingNeighbour, indexWeights);
                neighboursToConsider.emplace(dist, existingNeighbour);
            }

            // To do: use diversifiedCandidates heuristic, resize nodes and update neighbours with new selected neighbours
            */
        }
    }
}


vector<entity_id_t> MultiHNSW::internalSearch(const vector<span<const float>>& userQuery, size_t k, const vector<float>& weights) const {
    if (numEntities == 0) {
        return {};
    }

    // normalise query vectors if using cosine distance
    std::vector<std::span<const float>> query = userQuery;
    // storage for normalized query vectors to ensure valid spans
    // this storage must be defined outside the if block to ensure the data is in scope throughout the search
    std::vector<std::vector<float>> normalisedVectors;
    if (!toNormalise.empty()) {
        std::vector<std::span<const float>> normalisedQuery;
        for (size_t i = 0; i < numModalities; ++i) {
            if (distanceMetrics[i] == DistanceMetric::Cosine) {
                std::cout << "Normalising query for modality " << i << " to efficiently compute cosine distance" << std::endl;
                // copy and normalize vector
                normalisedVectors.emplace_back(query[i].begin(), query[i].end());
                l2NormalizeVector(std::span(normalisedVectors.back()));
                normalisedQuery.emplace_back(std::span(normalisedVectors.back()));
            } else {
                normalisedQuery.push_back(query[i]);
            }
        }
        query = std::move(normalisedQuery);
    }

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

[[nodiscard]] priority_queue<pair<float, entity_id_t>> MultiHNSW::searchLayer(const vector<span<const float>>& query, const vector<entity_id_t> &entryPoints, const vector<float>& weights, size_t ef, size_t layer) const {
    assert(!entryPoints.empty());
    assert(layer <= maxLevel);
    // set of visited elements initialised to entryPoints
    unordered_set<entity_id_t> visited(entryPoints.begin(), entryPoints.end());
    // min priority queue of candidates, stores negative dist
    priority_queue<pair<float, entity_id_t>> candidates;
    // fixed-size priority queue of nearest neighbours found so far (ef size)
    priority_queue<pair<float, entity_id_t>> nearestNeighbours;

    // populate the priority queues
    for (entity_id_t entryPoint : entryPoints) {
        const float dist = computeDistanceToQuery(entryPoint, query, weights);
        candidates.emplace(-dist, entryPoint);
        nearestNeighbours.emplace(dist, entryPoint);
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
        for (entity_id_t neighbour : nodes[bestCandidate].neighboursPerLayer[layer]) {
            if (!visited.contains(neighbour)) {
                visited.insert(neighbour);
                const float newDist = computeDistanceToQuery(neighbour, query, weights);
                auto [_worstNeighbourDist, _worstNeighbour] = nearestNeighbours.top();
                if (newDist < _worstNeighbourDist || nearestNeighbours.size() < ef) {
                    candidates.emplace(-newDist, neighbour);
                    nearestNeighbours.emplace(newDist, neighbour);
                    if (nearestNeighbours.size() > ef) {
                        nearestNeighbours.pop();
                    }
                }
            }
        }
    }
    return nearestNeighbours;
}

// Warning: this is duplicated code of above but just for searching an entity
[[nodiscard]] priority_queue<pair<float, entity_id_t>> MultiHNSW::searchLayer(entity_id_t entityId, const vector<entity_id_t> &entryPoints, const vector<float>& weights, size_t ef, size_t layer) const {
    assert(!entryPoints.empty());
    assert(layer <= maxLevel);
    // set of visited elements initialised to entryPoints
    unordered_set<entity_id_t> visited(entryPoints.begin(), entryPoints.end());
    // min priority queue of candidates, stores negative dist
    priority_queue<pair<float, entity_id_t>> candidates;
    // fixed-size priority queue of nearest neighbours found so far (ef size)
    priority_queue<pair<float, entity_id_t>> nearestNeighbours;

    // populate the priority queues
    for (entity_id_t entryPoint : entryPoints) {
        const float dist = computeDistanceBetweenEntities(entryPoint, entityId, weights);
        //const float dist = computeDistanceToQuery(entryPoint, query, weights);
        candidates.emplace(-dist, entryPoint);
        nearestNeighbours.emplace(dist, entryPoint);
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
        for (entity_id_t neighbour : nodes[bestCandidate].neighboursPerLayer[layer]) {
            if (!visited.contains(neighbour)) {
                visited.insert(neighbour);
                const float newDist = computeDistanceBetweenEntities(neighbour, entityId, weights);
                //const float newDist = computeDistanceToQuery(neighbour, query, weights);
                auto [_worstNeighbourDist, _worstNeighbour] = nearestNeighbours.top();
                if (newDist < _worstNeighbourDist || nearestNeighbours.size() < ef) {
                    candidates.emplace(-newDist, neighbour);
                    nearestNeighbours.emplace(newDist, neighbour);
                    if (nearestNeighbours.size() > ef) {
                        nearestNeighbours.pop();
                    }
                }
            }
        }
    }
    return nearestNeighbours;
}


void MultiHNSW::selectNearestCandidates(priority_queue<pair<float, entity_id_t>> &candidates, size_t M) const {
    // precondition: candidates is a non-empty max heap
    assert(!candidates.empty());
    assert(candidates.top().first >= 0);

    // pop from the heap until needed
    while (candidates.size() > M) {
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
priority_queue<pair<float, entity_id_t>> MultiHNSW::selectNearestCandidates(entity_id_t targetEntityId, const span<entity_id_t> candidates, size_t M, const std::vector<float>& weights) const {

    // precondition: candidates is a non-empty array
    assert(!candidates.empty());
    assert(M <= candidates.size());
    priority_queue<pair<float, entity_id_t>> maxHeap;
    for (const entity_id_t candidate : candidates) {
        assert(candidate != targetEntityId);
        const float dist = computeDistanceBetweenEntities(targetEntityId, candidate, weights);
        if (maxHeap.size() < M) {
            maxHeap.emplace(dist, candidate);
        } else if (dist < maxHeap.top().first) {
            maxHeap.pop();
            maxHeap.emplace(dist, candidate);
        }
    }
    assert(!maxHeap.empty() && maxHeap.size() <= M);
    return maxHeap;
}


// selectDiversifiedCandidates

void MultiHNSW::printGraph() const {
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

void MultiHNSW::save(const string& path) const {
    cout << "Saving MultiHNSW to " << path << endl;
}

void MultiHNSW::load(const string& path) {
    cout << "Loading index from " << path << endl;
}

//implement getters and setters
float MultiHNSW::getDistributionScaleFactor() const {
    return distributionScaleFactor;
}

size_t MultiHNSW::getTargetDegree() const {
    return targetDegree;
}

size_t MultiHNSW::getMaxDegree() const {
    return maxDegree;
}

size_t MultiHNSW::getEfConstruction() const {
    return efConstruction;
}

size_t MultiHNSW::getEfSearch() const {
    return efSearch;
}

size_t MultiHNSW::getSeed() const {
    return seed;
}

void MultiHNSW::setEfSearch(size_t efSearch) {
    if (efSearch < 1) {
        throw invalid_argument("efSearch must be at least 1");
    }
    this->efSearch = efSearch;
}