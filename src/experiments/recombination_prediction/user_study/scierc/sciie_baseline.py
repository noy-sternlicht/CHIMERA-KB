import argparse
import os

import pandas as pd
import torch
from util import setup_default_logger
from lputil import get_entities_data, get_eval_samples, build_eval_dataset, load_model, MODELS
from tqdm import tqdm

def main():
    model = load_model(32, False, '', args.model_name)
    eval_data = build_eval_dataset(args.test_path, args.model_name)
    logger.info(f'Loaded {len(eval_data)} eval samples from {args.test_path}')

    entities_data = get_entities_data(args.entities_path)
    logger.info(f'Loaded {len(entities_data)} entities from {args.entities_path}')
    eval_samples, _ = get_eval_samples(args.all_edges_path, args.test_candidates_path, eval_data,
                                       entities_data)
    eval_samples = [sample for sample in eval_samples if sample['relation'] == 'inspiration']
    logger.info(f'Loaded {len(eval_samples)} eval samples')

    sciie_entities = pd.read_csv(args.sciie_entities, names=['entity_text'])
    entities_ids = sciie_entities.index.tolist()
    entities_texts = sciie_entities['entity_text'].tolist()
    entity_id_to_text = {i: entities_texts[i] for i in entities_ids}

    model_info = MODELS[args.model_name]

    model = model.eval()
    with torch.no_grad():
        logger.info(f'Encoding {len(eval_samples)} queries...')
        queries = [s['query'] for s in eval_samples]
        if model_info['query_prompt']:
            logger.info(f'Using query prompt: {model_info["query_prompt"]}')
            queries_encoded = model.encode(queries, batch_size=1024,
                                           prompt=model_info['query_prompt'])
            torch.cuda.empty_cache()
        else:
            queries_encoded = model.encode(queries, batch_size=1024)
            torch.cuda.empty_cache()

        logger.info(f'Encoding {len(entities_texts)} entities...')
        sorted_ids = sorted(entities_ids)
        entities_sorted = [entity_id_to_text[i] for i in sorted_ids]
        if model_info['answer_prompt']:
            logger.info(f'Using answer prompt: {model_info["answer_prompt"]}')
            entities_encoded = model.encode(entities_sorted, batch_size=1024,
                                            prompt=model_info['answer_prompt'])
            torch.cuda.empty_cache()
        else:
            entities_encoded = model.encode(entities_sorted, batch_size=1024)
            torch.cuda.empty_cache()
        entity_id_to_encoded = {e: entities_encoded[i,] for i, e in enumerate(sorted_ids)}

    results = []
    for i, sample in enumerate(tqdm(eval_samples)):
        candidates_encoded = [entity_id_to_encoded[e] for e in entities_ids]
        similarities = model.similarity(queries_encoded[i,], candidates_encoded)
        ranked_candidates_indices = torch.argsort(similarities, descending=True).tolist()[0]

        results.append({
            'id': sample['id'],
            'query': sample['query'],
            'context': sample['context'],
            'anchor_id': sample['anchor_id'],
            'anchor_text': sample['anchor_text'],
            'relation': sample['relation'],
            'positive': sample['positive'],
            'is_cross_domain': sample['is_cross_domain'],
            'candidates': [entities_ids[idx] for idx in ranked_candidates_indices],
            'candidates_text': [entities_texts[idx] for idx in ranked_candidates_indices]
        })

    results_path = os.path.join(args.output_path, 'llm_baseline_results.json')
    results_df = pd.DataFrame(results)
    results_df.to_json(results_path, orient='records', lines=True)
    logger.info(f'Saved LLM baseline results to {results_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--test_path', type=str)
    parser.add_argument('--entities_path', type=str)
    parser.add_argument('--all_edges_path', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--sciie_entities', type=str)
    parser.add_argument('--test_candidates_path', type=str)

    args = parser.parse_args()

    logger = setup_default_logger(args.output_path)

    main()
