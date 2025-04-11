import argparse
import os
from typing import List

from util import setup_default_logger
import json
from lputil import load_model
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd

def cluster_entities(inputs: List[str], threshold: float, model: SentenceTransformer):
    if len(inputs) < 2:
        return inputs
    logger.info(f"Encoding {len(inputs)} inputs")
    embeddings = model.encode(inputs, batch_size=1024, show_progress_bar=True)

    logger.info(f"Clustering {len(inputs)} embeddings")
    clusters = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, linkage='average',
                                       metric="cosine").fit(embeddings)

    logger.info(f"Grouping inputs by cluster")
    input_by_cluster = {}
    for i, cluster_id in enumerate(clusters.labels_):
        if cluster_id not in input_by_cluster:
            input_by_cluster[cluster_id] = []
        input_by_cluster[cluster_id].append(inputs[i])

    logger.info(f"Selecting cluster representatives")
    clusters_representatives = []
    for cluster_id, cluster_strings in input_by_cluster.items():
        cluster_embeddings = np.array(model.encode(cluster_strings))
        centroid = np.mean(cluster_embeddings, axis=0)
        closest_index = np.argmin([np.dot(centroid, embedding) /
                                   (np.linalg.norm(centroid) * np.linalg.norm(embedding))
                                   for embedding in cluster_embeddings])
        clusters_representatives.append(cluster_strings[closest_index])
    return clusters_representatives


def main():
    entities_by_rel_type = json.load(open(args.data_path))
    selected_relations = args.selected_relations

    entities_by_rel_type_filtered = {}
    for rel_type, entities in entities_by_rel_type.items():
        if rel_type in selected_relations:
            entities_by_rel_type_filtered[rel_type] = entities_by_rel_type[rel_type]

    logger.info(f'Loading model {args.encoder}')
    model = load_model(32, False, '', args.encoder).to('cuda')
    all_entities = []
    for entities in entities_by_rel_type_filtered.values():
        all_entities.extend(entities)

    all_entities = list(set(all_entities))

    reduced_entities = cluster_entities(all_entities, args.clustering_thershold, model)
    logger.info(f"Reduced entities from {len(all_entities)} to {len(reduced_entities)}")
    out_path = os.path.join(args.output_path, 'reduced_entities.csv')
    pd.DataFrame(reduced_entities).to_csv(out_path, index=False)
    logger.info(f"Reduced entities saved to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--encoder', type=str, required=True)
    parser.add_argument('--clustering_thershold', type=float, default=0.1)
    parser.add_argument('--selected_relations', nargs='+', required=True)

    args = parser.parse_args()
    logger = setup_default_logger(args.output_path)

    main()
