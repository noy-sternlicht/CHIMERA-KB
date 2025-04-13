import argparse
import json

import torch
from lputil import get_entities_data, load_model, build_query, read_edges
from util import setup_default_logger
from sentence_transformers import util
from rank_gpt import sliding_windows


def build_reranker_query(anchor_text, relation):
    if relation == 'combination':
        query = f'What could we blend with "{anchor_text}" to address the described settings?'
    else:
        anchor_text = anchor_text.capitalize()
        query = f'What would be a good source of inspiration for "{anchor_text}"?'

    return query


def rank_gpt_reranker(query, candidates_text, candidates_ids):
    logger.info(f'Reranking {len(candidates_text)} candidates for using GPT')
    reranker_input = {'query': query,
                      'hits': [{'id': candidate_id, 'content': candidate_text} for candidate_id, candidate_text
                               in zip(candidates_ids, candidates_text)]}
    api_key = open('openai_key', 'r').read().strip()

    nr_reranked = len(candidates_text)
    reranker_output = sliding_windows(reranker_input, rank_start=0, rank_end=nr_reranked,
                                      window_size=20,
                                      step=5,
                                      model_name='gpt-4o',
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


def main():
    input = json.load(open(args.input_path))
    query = build_query(input['context'], input['anchor'], input['recombination_type'])

    entities_data = get_entities_data(args.entities_path)
    candidates = []
    with open(args.test_candidates_path, 'r') as f:
        for line in f:
            candidates.append(line.strip())
    entity_id_to_text = {e: entities_data[e] for e in candidates}

    candidates = list(set(candidates))
    candidates_text = [entity_id_to_text[e] for e in candidates]

    model = load_model(args.weights_precision, args.quantize, args.checkpoint, args.model_name)
    query_encoded = model.encode(query, show_progress_bar=True, batch_size=1024)
    entities_encoded = model.encode(candidates_text, show_progress_bar=True, batch_size=1024)

    response = util.semantic_search(query_encoded, entities_encoded, score_function=util.cos_sim, top_k=20)[0]
    ranker_response_text = [candidates_text[hit['corpus_id']] for hit in response]
    ranker_response_ids = [candidates[hit['corpus_id']] for hit in response]


    reranker_query = build_reranker_query(input['anchor'], input['recombination_type'])
    reranker_query = input['context'] + '\n' + reranker_query
    reranker_response = rank_gpt_reranker(reranker_query, ranker_response_text, ranker_response_ids)

    print('============QUERY============')
    print(reranker_query)

    print('============TOP-10-SUGGESTIONS============')
    for i in range(10):
        print(f"{i + 1}. {reranker_response['reranked_texts'][i]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--entities_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--test_candidates_path', type=str)
    parser.add_argument('--quantize', action='store_true')
    parser.add_argument('--weights_precision', type=int, default=16)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--checkpoint', type=str, default=None)

    args = parser.parse_args()

    logger = setup_default_logger(args.output_path)

    main()
