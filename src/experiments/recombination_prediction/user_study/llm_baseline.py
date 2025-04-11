import argparse

import pandas as pd
from lputil import get_entities_data, get_eval_samples, build_eval_dataset

from util import setup_default_logger, init_openai_client, request_openai_batch_completions, get_openai_batch_completions
import json
import time
import os

PROMPT_TEMPLATE = """You are an AI assistant tasked with suggesting a concise answer for a scientific query requesting a recommendation. Your goal is to provide a brief, focused response based on the given query.

Here is the scientific query:
<query>
{{QUERY}}
</query>

Your answer should be extremely concise and shorter than a sentence. For example: "transformer models" or "the memory capacity of the human brain".

Provide your answer in the following format:

<answer>
[Your concise recommendation here]
</answer>

Remember to keep your answer brief and to the point. Do not include any additional reasoning, explanation, or justification for your recommendation. Simply state the most appropriate answer to the query."""


def build_query(context, anchor_text, relation):
    if relation == 'combination':
        query = f"What could we blend with **{anchor_text}** to address the context?"
    else:
        anchor_text = anchor_text.capitalize()
        query = f"What would be a good source of inspiration for **{anchor_text}**?"

    return f'{context}\n{query}'


def main():
    eval_data = build_eval_dataset(args.test_path, 'sentence-transformers/all-mpnet-base-v2')
    logger.info(f'Loaded {len(eval_data)} eval samples from {args.test_path}')
    entities_data = get_entities_data(args.entities_path)
    logger.info(f'Loaded {len(entities_data)} entities from {args.entities_path}')
    eval_samples, _ = get_eval_samples(args.all_edges_path, args.test_candidates_path, eval_data,
                                       entities_data)
    eval_samples = [sample for sample in eval_samples if sample['relation'] == 'inspiration']
    logger.info(f'Loaded {len(eval_samples)} eval samples')

    prompts = {}
    for sample in eval_samples:
        query = build_query(sample['context'], sample['anchor_text'], sample['relation'])
        prompt = PROMPT_TEMPLATE.replace('{{QUERY}}', query)
        prompts[sample['id']] = prompt

    batch_idx = request_openai_batch_completions(prompts, 50, 0.5, 0, args.output_path, OPEN_AI_CLIENT,
                                                 args.openai_engine)

    logger.info(f'Waiting for batch {batch_idx} to complete...')
    responses, status = get_openai_batch_completions(batch_idx, OPEN_AI_CLIENT)
    while status != 'completed':
        logger.info(f'Waiting for batch {batch_idx} to complete...')
        responses, status = get_openai_batch_completions(batch_idx, OPEN_AI_CLIENT)
        time.sleep(60)  # Wait for 1 minute

    logger.info(f'Batch {batch_idx} completed')
    responses_path = os.path.join(args.output_path, f'responses-{batch_idx}.json')
    with open(responses_path, 'w') as f:
        json.dump(responses, f)
    logger.info(f'Saved responses to {responses_path}')

    samples_by_id = {sample['id']: sample for sample in eval_samples}
    results = []
    for sample_id, response in responses.items():
        parsed_response = ''
        answer_start = response.find('<answer>')
        answer_end = response.find('</answer>')
        if answer_start != -1 and answer_end != -1:
            parsed_response = response[answer_start + len('<answer>'):answer_end].strip()
        else:
            logger.warning(f'No answer found in response: {response}')

        sample = samples_by_id[sample_id]

        results.append({
            'id': sample['id'],
            'query': sample['query'],
            'context': sample['context'],
            'anchor_id': sample['anchor_id'],
            'anchor_text': sample['anchor_text'],
            'relation': sample['relation'],
            'positive': parsed_response})

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
    parser.add_argument('--test_candidates_path', type=str)
    parser.add_argument('--openai_engine', type=str)

    args = parser.parse_args()
    OPEN_AI_CLIENT = init_openai_client()

    logger = setup_default_logger(args.output_path)

    main()
