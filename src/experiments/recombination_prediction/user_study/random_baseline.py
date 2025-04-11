import argparse
import os

import pandas as pd
from lputil import get_entities_data, get_eval_samples, build_eval_dataset
import random

from util import setup_default_logger

from tqdm import tqdm


def main():
    eval_data = build_eval_dataset(args.test_path, 'sentence-transformers/all-mpnet-base-v2')
    logger.info(f'Loaded {len(eval_data)} eval samples from {args.test_path}')
    entities_data = get_entities_data(args.entities_path)
    logger.info(f'Loaded {len(entities_data)} entities from {args.entities_path}')
    eval_samples, _ = get_eval_samples(args.all_edges_path, args.test_candidates_path, eval_data,
                                       entities_data)
    logger.info(f'Loaded {len(eval_samples)} eval samples')

    results = []
    pb = tqdm(total=len(eval_samples))
    for sample in eval_samples:
        candidate_id_to_text = {cand_id: text for cand_id, text in zip(sample['candidates'], sample['candidates_text'])}

        random.shuffle(sample['candidates'])
        candidates_text = [candidate_id_to_text[cand_id] for cand_id in sample['candidates']]
        pos_id = str(sample['positive_id'])
        sample['candidates'] = [str(cand) for cand in sample['candidates']]
        pos_rank = sample['candidates'].index(pos_id)

        results.append(
            {'id': sample['id'],
             'query': sample['query'],
             'context': sample['context'],
             'anchor_id': sample['anchor_id'],
             'anchor_text': sample['anchor_text'],
             'relation': sample['relation'],
             'positive': sample['positive'],
             'positive_id': sample['positive_id'],
             'is_cross_domain': sample['is_cross_domain'],
             'candidates': sample['candidates'],
             'candidates_text': candidates_text,
             'positive_index': pos_rank}
        )
        pb.update(1)

    results_path = os.path.join(args.output_path, 'random_baseline_results.json')
    results_df = pd.DataFrame(results)
    results_df.to_json(results_path, orient='records', lines=True)
    logger.info(f'Saved random baseline results to {results_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--test_path', type=str)
    parser.add_argument('--entities_path', type=str)
    parser.add_argument('--all_edges_path', type=str)
    parser.add_argument('--test_candidates_path', type=str)

    args = parser.parse_args()

    logger = setup_default_logger(args.output_path)

    main()
