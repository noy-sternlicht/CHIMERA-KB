import argparse
import ast

from util import setup_default_logger
import pandas as pd
import os
import json


def domain_comparison(row):
    source_part = row['source_domain'].split('.')[0]
    target_part = row['target_domain'].split('.')[0]

    return source_part != target_part


def main():
    baseline_results = json.load(open(args.baselines_results_path))['baselines']
    aggregated_results = {}

    for baseline in args.baselines_to_compare:
        logger.info(f'Processing baseline {baseline}')
        results_path = baseline_results[baseline]
        if results_path.endswith('.json'):
            results = pd.read_json(results_path, orient='records', lines=True)
        else:
            results = pd.read_csv(results_path)
            results['candidates_text'] = results['candidates_text'].apply(ast.literal_eval)
        for _, row in results.iterrows():
            if args.inspiration_examples_only and row['relation'] != 'inspiration':
                continue

            if row['id'] not in aggregated_results:
                aggregated_results[row['id']] = {
                    'context': row['context'],
                    'anchor': row['anchor_text'],
                    'relation': row['relation'],
                    'query': row['query'],
                    'positive': row['positive'],
                }
            if 'candidates_text' in row:
                assert isinstance(row['candidates_text'], list)
                aggregated_results[row['id']][baseline] = row['candidates_text'][:args.top_k]
            else:
                aggregated_results[row['id']][baseline] = [row['positive']] * args.top_k

    logger.info(f'Aggregated results for {len(aggregated_results)} queries')

    test_data = pd.read_csv(args.test_path)
    id_to_arxiv_categories = {row['id']: row['arxiv_categories'] for _, row in test_data.iterrows()}
    id_to_is_cross_domain = {row['id']: domain_comparison(row) for _, row in test_data.iterrows()}

    human_study_examples = []
    for query_id, data in aggregated_results.items():
        row_id = query_id.split('_')[1]
        arxiv_categories = id_to_arxiv_categories[row_id].split(',')

        for i in range(args.top_k):
            qa = {
                'context': data['context'],
                'anchor': data['anchor'],
                'relation': data['relation'],
                'query': data['query'],
                'k': i + 1,
                'positive': data['positive'],
                'id': f'{i + 1}-{query_id}',
                'arxiv_categories': arxiv_categories,
                'is_cross_domain': id_to_is_cross_domain[row_id]
            }
            missing_baseline = False
            for baseline in args.baselines_to_compare:
                if baseline not in data:
                    missing_baseline = True
                    break
                qa[baseline] = data[baseline][i]
            if not missing_baseline:
                human_study_examples.append(qa)

    logger.info(f'Generated {len(human_study_examples)} human study examples')

    out_csv_path = os.path.join(args.output_path, 'human_study_examples.csv')
    results_df = pd.DataFrame(human_study_examples)
    results_df.to_csv(out_csv_path, index=False)
    logger.info(f'Saved results to {out_csv_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--baselines_results_path', type=str)
    parser.add_argument('--baselines_to_compare', nargs='+')
    parser.add_argument('--test_path', type=str)
    parser.add_argument('--top_k', type=int)
    parser.add_argument('--inspiration_examples_only', action='store_true')

    args = parser.parse_args()
    logger = setup_default_logger(args.output_path)

    main()
