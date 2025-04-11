import argparse
import ast
from datetime import datetime

from util import setup_default_logger
import json
import os
import statistics
import pandas as pd
import random
from tqdm import tqdm
from rank_gpt import permutation_pipeline, sliding_windows


def build_query(anchor_text, relation):
    if relation == 'combination':
        query = f'What could we blend with "{anchor_text}" to address the described settings?'
    else:
        anchor_text = anchor_text.capitalize()
        query = f'What would be a good source of inspiration for "{anchor_text}"?'

    return query


def get_ranking_str(ranked_candidates, ranked_candidates_idx=None, reasoning_chains=None):
    ranking_str = ''
    for i, candidate in enumerate(ranked_candidates):
        candidate_idx = f"[{ranked_candidates_idx[i]}]" if ranked_candidates_idx is not None else ''
        reasoning_chain = f'\nReasoning chain: {reasoning_chains[i]}' if reasoning_chains is not None else ''
        ranking_str += f'{i + 1}. {candidate}{candidate_idx}{reasoning_chain}\n'
    return ranking_str


def rank_gpt_reranker(query, candidates_text, candidates_ids):
    logger.info(f'Reranking {len(candidates_text)} candidates for using GPT')
    reranker_input = {'query': query,
                      'hits': [{'id': candidate_id, 'content': candidate_text} for candidate_id, candidate_text
                               in zip(candidates_ids, candidates_text)]}
    api_key = open('openai_key', 'r').read().strip()

    nr_reranked = len(candidates_text)
    if nr_reranked <= args.rank_gpt_window_size:
        reranker_output = permutation_pipeline(reranker_input, rank_start=0, rank_end=nr_reranked,
                                               model_name=args.openai_engine,
                                               api_key=api_key)
    else:
        reranker_output = sliding_windows(reranker_input, rank_start=0, rank_end=nr_reranked,
                                          window_size=args.rank_gpt_window_size,
                                          step=args.rank_gpt_step_size,
                                          model_name=args.openai_engine,
                                          api_key=api_key)
    out_ids = [hit['id'] for hit in reranker_output['hits']]
    for candidate_id in candidates_ids:
        if candidate_id not in out_ids:
            logger.warning(f'Candidate {candidate_id} not in reranker output: {out_ids}')
    for out_id in out_ids:
        if out_id not in candidates_ids:
            logger.warning(f'Candidate {out_id} in reranker output but not in candidates: {candidates_ids}')

    return {'reranked_texts': [hit['content'] for hit in reranker_output['hits']],
            'reranked_ids': [hit['id'] for hit in reranker_output['hits']],
            'og_texts': candidates_text, 'og_ids': candidates_ids,
            'query': query}


def create_prompt(query, document):
    return (
        "Determine if the following passage is relevant to the query. "
        "Answer only with 'true' or 'false'.\n"
        f"Query: {query}\n"
        f"Passage: {document}\n"
        "<think>"
    )



def main():
    results_path = args.biencoder_results
    logger.info(f'Loading ranker results from {results_path}')
    if results_path.endswith('.json'):
        ranker_results = pd.read_json(results_path, orient='records', lines=True)
    else:
        ranker_results = pd.read_csv(results_path)
        ranker_results['candidates_text'] = ranker_results['candidates_text'].apply(ast.literal_eval)

    logger.info(f'Loaded {len(ranker_results)} ranker results from {args.biencoder_results}')

    metrics = {}
    filter_types = ['combination', 'inspiration', 'cross-domain', 'all']
    metric_types = ['mr', 'mrr', 'hits@1', 'hits@3', 'hits@5', 'hits@10', 'hits@50', 'hits@100', 'median_gold_rank']
    for ftype in filter_types:
        for metric in metric_types:
            if metric != 'median_gold_rank':
                metrics[f'{ftype}_{metric}'] = 0
            else:
                metrics[f'{ftype}_{metric}'] = []

    def update_metrics(metrics, rank, ftype):
        metrics[f'{ftype}_mr'] += rank
        metrics[f'{ftype}_mrr'] += 1 / rank
        metrics[f'{ftype}_hits@1'] += 1 if rank <= 1 else 0
        metrics[f'{ftype}_hits@3'] += 1 if rank <= 3 else 0
        metrics[f'{ftype}_hits@5'] += 1 if rank <= 5 else 0
        metrics[f'{ftype}_hits@10'] += 1 if rank <= 10 else 0
        metrics[f'{ftype}_hits@50'] += 1 if rank <= 50 else 0
        metrics[f'{ftype}_hits@100'] += 1 if rank <= 100 else 0
        metrics[f'{ftype}_median_gold_rank'].append(rank)

    responses_by_row_id = {}
    pbar = tqdm(total=len(ranker_results))
    row_num = 0
    for _, row in ranker_results.iterrows():
        pbar.update(1)
        query = build_query(row['anchor_text'], row['relation'])
        query = row['context'] + '\n' + query
        candidates_text = row['candidates_text'][:args.top_k].copy()
        candidate_ids = row['candidates'][:args.top_k].copy()
        old_gold_rank = row['positive_index'] + 1

        if row_num % args.perform_checkpoint_at == 0:
            logger.info(f'Checkpoint at idx={row_num}')
            responses_path = os.path.join(args.output_dir, 'responses_by_row_id.json')
            with open(responses_path, 'w') as f:
                json.dump(responses_by_row_id, f)
            logger.info(f'Wrote responses by row id to {responses_path}')
        row_num += 1

        if old_gold_rank > args.top_k or old_gold_rank > len(row['candidates']):
            if args.always_rerank:
                random_idx = random.randint(0, len(candidates_text) - 1)
                candidates_text = candidates_text[:random_idx] + [row['positive']] + candidates_text[random_idx:]
                candidate_ids = candidate_ids[:random_idx] + [row['positive_id']] + candidate_ids[random_idx:]
            else:
                continue

        responses_by_row_id[row['id']] = rank_gpt_reranker(query, candidates_text, candidate_ids)

    responses_path = os.path.join(args.output_dir, 'responses_by_row_id.json')
    with open(responses_path, 'w') as f:
        json.dump(responses_by_row_id, f)
    logger.info(f'Wrote responses by row id to {responses_path}')

    times_worse, times_better, placed_worse, places_better, nr_reranked = 0, 0, 0, 0, 0
    readable_results = ''
    new_ranks = pd.DataFrame()
    output_path = os.path.join(args.output_dir, 'reranked_results.json')
    for idx, row in ranker_results.iterrows():
        old_gold_rank = row['positive_index'] + 1
        row_copy = row.copy()

        if not args.always_rerank and (old_gold_rank > args.top_k or old_gold_rank > len(row['candidates'])):
            gold_rank = old_gold_rank
        else:
            pos_text = row['positive']
            pos_id = str(row['positive_id'])
            og_candidates_texts = responses_by_row_id[row['id']]['og_texts']
            og_candidates_ids = responses_by_row_id[row['id']]['og_ids']
            og_candidates_ids = [str(cid) for cid in og_candidates_ids]
            old_gold_rank = og_candidates_ids.index(pos_id) + 1
            reranked_candidates_texts = responses_by_row_id[row['id']]['reranked_texts']
            reranked_candidates_ids = responses_by_row_id[row['id']]['reranked_ids']
            reranked_candidates_ids = [str(cid) for cid in reranked_candidates_ids]

            row_results_str = f'\n==========={row["id"]}============\n'
            row_results_str += f'---------Query---------\n{responses_by_row_id[row["id"]]["query"]}\n'
            row_results_str += f'---------OG-Ranking---------\n{get_ranking_str(og_candidates_texts)}\n'
            row_results_str += f'---------Reranked-Ranking---------\n{get_ranking_str(reranked_candidates_texts)}\n'

            row_copy['candidates'] = reranked_candidates_ids + row_copy['candidates'][args.top_k:]
            row_copy['candidates'] = [str(cid) for cid in row_copy['candidates']]
            row_copy['candidates_text'] = reranked_candidates_texts + row_copy['candidates_text'][args.top_k:]
            if pos_id not in reranked_candidates_ids:
                logger.warning(
                    f'[{row["id"]}]: Gold candidate "{pos_id}-{pos_text}" not in reranked candidates:\n{get_ranking_str(reranked_candidates_texts)}, og rank: {old_gold_rank}')
                gold_rank = old_gold_rank
            else:
                gold_rank = reranked_candidates_ids.index(pos_id) + 1
                row_copy['positive_index'] = gold_rank - 1
                nr_reranked += 1
                row_results_str += f'---------Gold---------\nText: "{pos_text}"\nOld-rank: {old_gold_rank}\nNew-rank: {gold_rank}\n'
                readable_results += row_results_str
                logger.info(row_results_str)

            if old_gold_rank > gold_rank:
                logger.info(f'Row {row["id"]}: improved rank from {old_gold_rank} to {gold_rank}')
                times_better += 1
                places_better += old_gold_rank - gold_rank
            elif old_gold_rank < gold_rank:
                logger.info(f'Row {row["id"]}: worsened rank from {old_gold_rank} to {gold_rank}')
                times_worse += 1
                placed_worse += gold_rank - old_gold_rank

        new_ranks = pd.concat([new_ranks, row_copy.to_frame().T])

        update_metrics(metrics, gold_rank, 'all')
        if row['is_cross_domain']:
            update_metrics(metrics, gold_rank, 'cross-domain')
        if row['relation'] == 'combination':
            update_metrics(metrics, gold_rank, 'combination')
        if row['relation'] == 'inspiration':
            update_metrics(metrics, gold_rank, 'inspiration')

    for ftype in filter_types:
        nr_queries = len(metrics[f'{ftype}_median_gold_rank'])
        logger.info(f'Number of queries for {ftype}: {nr_queries}')
        for metric in metric_types:
            if metric == 'median_gold_rank':
                if len(metrics[f'{ftype}_{metric}']) == 0:
                    metrics[f'{ftype}_{metric}'] = 0
                else:
                    metrics[f'{ftype}_{metric}'] = statistics.median(metrics[f'{ftype}_{metric}'])
            elif nr_queries > 0:
                metrics[f'{ftype}_{metric}'] /= nr_queries
            else:
                metrics[f'{ftype}_{metric}'] = 0

    logger.info(f'Metrics: {json.dumps(metrics, indent=2)}')
    if nr_reranked > 0 and times_better > 0 and times_worse > 0:
        logger.info(
            f'Improved rank {times_better} / {nr_reranked} times: {times_better / nr_reranked}, places better: {places_better / times_better}.')
        logger.info(
            f'Worsened rank {times_worse} / {nr_reranked} times: {times_worse / nr_reranked}, places worse: {placed_worse / times_worse}.')

    new_ranks.to_json(output_path, orient='records', lines=True)
    logger.info(f'Wrote reranked results to {output_path}')

    readable_results_path = os.path.join(args.output_dir, 'readable_results.log')
    with open(readable_results_path, 'w') as f:
        f.write(readable_results)

    logger.info(f'Wrote readable results to {readable_results_path}')

    table_log = '\n & H@1 & H@3 & H@5 & H@10 & H@50 & H@100 & MRR & Median Rank \\\\ \n'
    for ftype in filter_types:
        table_log += f'----{ftype}\n'
        table_log += f' & {metrics[f"{ftype}_hits@1"]:.3f} & {metrics[f"{ftype}_hits@3"]:.3f} & {metrics[f"{ftype}_hits@5"]:.3f} & {metrics[f"{ftype}_hits@10"]:.3f} & {metrics[f"{ftype}_hits@50"]:.3f} & {metrics[f"{ftype}_hits@100"]:.3f} & {metrics[f"{ftype}_mrr"]:.3f} & {metrics[f"{ftype}_median_gold_rank"]:.3f} \\\\ \n'

    table_log_path = os.path.join(args.output_dir, 'table.log')
    with open(table_log_path, 'w') as f:
        f.write(table_log)

    logger.info(f'Table log:\n{table_log}')
    logger.info(f'Wrote table log to {table_log_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--biencoder_results', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--openai_engine', type=str, required=True)
    parser.add_argument('--top_k', type=int)
    parser.add_argument('--rank_gpt_window_size', type=int)
    parser.add_argument('--rank_gpt_step_size', type=int)
    parser.add_argument('--always_rerank', action='store_true')
    parser.add_argument('--perform_checkpoint_at', type=int)

    args = parser.parse_args()

    output_dir = args.output_dir
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    model_name = ''
    model_name += args.openai_engine

    args.output_dir = os.path.join(output_dir, f'{model_name}_{timestamp}')

    logger = setup_default_logger(args.output_dir)

    main()
