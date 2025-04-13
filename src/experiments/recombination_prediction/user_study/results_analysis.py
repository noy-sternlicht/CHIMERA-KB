import argparse
import json
import pandas as pd


def main():
    records = pd.read_csv(args.responses_path)
    baselines_results = []
    support = 0

    for _, record in records.iterrows():
        record_baselines_results = json.loads(record['baselines_results'])
        if not record['is_ill_defined']:
            support += 1

        knowledge_level = record['knowledge_level']

        for baseline, results in record_baselines_results.items():
            baselines_results.append({
                'example_id': record['example_id'],
                'annotator': record['annotator'],
                'context': record['context'],
                'query': record['query'],
                'gold': record['gold'],
                'baseline': baseline,
                'suggestion': results['suggestion'],
                'is_ill_defined': record['is_ill_defined'] == 'checked',
                'knowledge_level': knowledge_level,
                'k': results['k'],
                'rank': results['rank'],
                'normalized_rank': (1 / 6) * float(knowledge_level) * float(results['rank'])
            })

    baselines_results_df = pd.DataFrame(baselines_results)

    baselines_results_df = baselines_results_df[~baselines_results_df['is_ill_defined']]

    baselines_results_df.to_csv('baselines_results.csv', index=False)
    print('Saved baselines results to baselines_results.csv')

    scores = baselines_results_df.groupby('baseline')['rank']
    print(scores.describe(percentiles=[0.2, 0.4, 0.6, 0.8]).round(2))

    results = pd.DataFrame()
    results['meanRank'] = baselines_results_df.groupby('baseline')['rank'].mean().round(2)
    results['meanRankNorm'] = baselines_results_df.groupby('baseline')['normalized_rank'].mean().round(2)
    results['medianRank'] = baselines_results_df.groupby('baseline')['rank'].median().round(2)
    results['medianRankNorm'] = baselines_results_df.groupby('baseline')['normalized_rank'].median().round(2)

    print(results)
    print(f'Support: {support}')

    results_summary_path = 'baselines_results_summary.csv'
    results.to_csv(results_summary_path)
    print(f'Saved baselines results summary to {results_summary_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--responses_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)

    args = parser.parse_args()

    main()
