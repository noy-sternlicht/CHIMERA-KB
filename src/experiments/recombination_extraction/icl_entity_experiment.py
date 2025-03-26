import argparse
import json
import os
import re
from typing import List, Dict
import time
import pandas as pd
from util import setup_default_logger, get_openai_batch_completions, request_openai_batch_completions, \
    init_openai_client, NER_ENTITY_TYPES_ATTRIBUTES
from eval import RelationEntity, compute_entity_agreement

PROMPT_NER = """You are an AI assistant tasked with identifying specific types of entities in a given scientific abstract. Your goal is to carefully analyze the text and categorize entities according to the provided types.

Start by reviewing examples to understand the expected output format and the process of categorizing entities accurately. Use these examples as a reference throughout the task:  
{EXAMPLES}

Carefully read the text provided below and identify relevant entities:  
<input_to_analyze>
{TEXT}
</input_to_analyze>

Structure your identified entities in a JSON format using entity types as keys and lists of entities as values. Reference the initial examples for format guidance. If no entities of the specified types are present, provide an empty JSON object: {}

Ensure that your JSON output is valid:
- Use double quotes around strings
- Do not include a trailing comma after the last item in a list or object
- Escape any double quotes that appear within entity names

Finally, enclose your JSON output in <output_json> tags to denote completion like so: <output_json>{JSON_OUTPUT}</output_json>
"""


def prep_prompt_template(icl_examples: pd.DataFrame, entity_types: List[str]) -> str:
    prompt_template = PROMPT_NER
    examples_text = ''
    example_idx = 1
    for _, example in icl_examples.iterrows():
        text = example['text']
        entities = example['entities']
        example_text = f"Example {example_idx}:\n<input>\n{text}\n</input>\n<output>\n{json.dumps(entities)}\n<output>\n\n"
        examples_text += example_text
        example_idx += 1

    prompt_template = prompt_template.replace('{EXAMPLES}', examples_text)

    entity_types_description = ''
    entity_type_examples = {}

    example_idx = 1
    for i, type_attributes in enumerate(NER_ENTITY_TYPES_ATTRIBUTES):
        if type_attributes['entity_type'] not in entity_types:
            continue
        entity_type_name = type_attributes['entity_type']
        if 'prompt_type_name' in type_attributes:
            entity_type_name = type_attributes['prompt_type_name']

        if type_attributes['entity_type'] in entity_types:
            entity_types_description += f"{i + 1}. {entity_type_name}: {type_attributes['desc']}\n"

        entity_type_examples[entity_type_name] = []
        for _ in range(type_attributes['nr_entities_in_example']):
            entity_type_examples[entity_type_name].append(f"entity{example_idx}")
            example_idx += 1

    prompt_template = prompt_template.replace('{ENTITY_TYPE_DESCRIPTIONS}', entity_types_description)

    return prompt_template


def parse_output(raw_out: str, entity_types: List[str]) -> Dict:
    pattern = '<output_json>(.*?)</output_json>'
    matches = re.findall(pattern, raw_out, re.DOTALL)
    if not matches:
        pattern = '{(.*?)}'
        matches = re.findall(pattern, raw_out, re.DOTALL)
        matches = ['{' + match.strip() + '}' for match in matches]
    if not matches:
        return {}
    for match in matches:
        raw_out = match.strip()
        logger.info(f"------Parsing JSON from: {raw_out}")
        try:
            parsed_answer = json.loads(raw_out)
        except json.decoder.JSONDecodeError:
            logger.info(f"Failed to parse JSON from: {raw_out}")
            return {}

        if isinstance(parsed_answer, str):
            parsed_answer = json.loads(parsed_answer)

        final_answer = {entity_type: parsed_answer[entity_type] for entity_type in entity_types if
                        entity_type in parsed_answer}
        return final_answer


def get_icl_set(icl_examples: pd.DataFrame, nr_samples_per_class: int):
    recombination_examples = icl_examples[icl_examples['document_class'] == 'relevant']
    inspiration_examples = []
    combination_examples = []
    for _, example in recombination_examples.iterrows():
        entities = json.loads(example['readable_entities'])
        if 'analogy-src' in entities:
            entities['inspiration-src'] = entities.pop('analogy-src')
            entities['inspiration-target'] = entities.pop('analogy-target')
            inspiration_examples.append({'text': example['text'], 'entities': entities})
        elif 'comb-element' in entities:
            combination_examples.append({'text': example['text'], 'entities': entities})

    inspiration_examples = pd.DataFrame(inspiration_examples)
    combination_examples = pd.DataFrame(combination_examples)

    irrelevant_examples = icl_examples[icl_examples['document_class'] == 'irrelevant']
    no_recombination_examples = []
    for _, example in irrelevant_examples.iterrows():
        no_recombination_examples.append({'text': example['text'], 'entities': {}})

    no_recombination_examples = pd.DataFrame(no_recombination_examples)

    selected_icl_examples = pd.DataFrame()
    all_examples_by_class = {'inspiration': inspiration_examples, 'combination': combination_examples,
                             'no_recombination': no_recombination_examples}
    for class_name, examples in all_examples_by_class.items():
        selected_examples = examples.sample(n=nr_samples_per_class)
        selected_icl_examples = pd.concat([selected_icl_examples, selected_examples])

    selected_icl_examples = selected_icl_examples.sample(frac=1).reset_index(drop=True)
    return selected_icl_examples


def load_data(icl_examples_path: str, eval_examples_path: str):
    icl_examples = pd.read_csv(icl_examples_path, dtype={'paper_id': str})
    logger.info(f"Loaded {len(icl_examples)} ICL examples from {icl_examples_path}")
    test_examples = pd.read_csv(eval_examples_path, dtype={'paper_id': str})
    logger.info(f"Loaded {len(test_examples)} test examples from {eval_examples_path}")
    return icl_examples, test_examples


def main(icl_examples_path: str, eval_examples_path: str, entity_types: List[str], nr_samples_per_class: int,
         nr_repeats: int):
    icl_examples, test_examples = load_data(icl_examples_path, eval_examples_path)

    results = {'comb-element': [], 'inspiration-src': [], 'inspiration-target': []}
    fixed_entity_types = ['inspiration-src', 'inspiration-target', 'comb-element']
    batches = {}
    for i in range(nr_repeats):
        logger.info(f"Repeat {i + 1} of {nr_repeats}")
        selected_icl_examples = get_icl_set(icl_examples, nr_samples_per_class)
        prompt_template = prep_prompt_template(selected_icl_examples, entity_types)

        entity_types_fixed = []
        for entity_type in entity_types:
            if entity_type == 'analogy-src':
                entity_types_fixed.append('inspiration-src')
            elif entity_type == 'analogy-target':
                entity_types_fixed.append('inspiration-target')
            else:
                entity_types_fixed.append(entity_type)
        entity_types = entity_types_fixed
        logger.info(f"Entity types to evaluate: {entity_types}")

        prompts = {}
        texts = {}
        gold_outs = {}
        for _, example in test_examples.iterrows():
            text = example['text']
            gold_output = json.loads(example['readable_entities'])
            adjusted_types = {'analogy-src': 'inspiration-src', 'analogy-target': 'inspiration-target'}
            for old_type, new_type in adjusted_types.items():
                if old_type in gold_output:
                    gold_output[new_type] = gold_output.pop(old_type)

            prompt = prompt_template.replace('{TEXT}', text)
            prompts[example['paper_id']] = prompt
            texts[example['paper_id']] = text
            gold_outs[example['paper_id']] = gold_output

        logger.info(f"Requesting completions for {len(prompts)} examples")
        batch_idx = request_openai_batch_completions(prompts, 2048, 0, i, args.output_dir, OPEN_AI_CLIENT,
                                                     args.openai_engine)
        batches[batch_idx] = {'prompts': prompts, 'texts': texts, 'gold_outs': gold_outs}

    batches_info_file = os.path.join(args.output_dir, 'batches_info.json')
    with open(batches_info_file, 'w') as f:
        json.dump(batches, f)
    logger.info(f"Saved batch info to {batches_info_file}")

    logger.info(f"Waiting for {nr_repeats} batches to finish: {list(batches.keys())}")

    finished_batches = {batch_id: False for batch_id in batches.keys()}

    while not all(finished_batches.values()):
        for batch_id, finished in finished_batches.items():
            if not finished:
                query_responses, batch_status = get_openai_batch_completions(batch_id, OPEN_AI_CLIENT)
                if query_responses:
                    batches[batch_id]['responses'] = query_responses
                    finished_batches[batch_id] = True
                    logger.info(f"Batch {batch_id} finished")
                else:
                    logger.info(f"Batch {batch_id} still running: {batch_status}")
        time.sleep(60)

    logger.info('All batches finished')

    for batch_id, batch_data in batches.items():
        pred_entities = []
        gold_entities = []
        texts = []
        for paper_id, response in batch_data['responses'].items():
            doc_gold_entities = []
            for entity_type, entities in batch_data['gold_outs'][paper_id].items():
                for entity in entities:
                    doc_gold_entities.append(RelationEntity('', entity_type, entity, []))

            result = parse_output(response, fixed_entity_types)
            doc_pred_entities = []
            for entity_type, entities in result.items():
                for entity in entities:
                    doc_pred_entities.append(RelationEntity('', entity_type, entity, []))

            log_string = f'------{paper_id}------\n'
            log_string += f"------TEXT------\n{batch_data['texts'][paper_id]}\n"
            log_string += f"------OUTPUT------\n{result}\n"

            texts.append(batch_data['texts'][paper_id])
            pred_entities.append(doc_pred_entities)
            gold_entities.append(doc_gold_entities)
            logger.info(log_string)

        iteration_results = compute_entity_agreement(pred_entities, gold_entities, texts,
                                                     fixed_entity_types, logger,
                                                     undefined_val=0)
        for entity_type in fixed_entity_types:
            # logger.info(f"Results for {entity_type}:\n{iteration_results[entity_type]}\n")
            results[entity_type].append(iteration_results[entity_type])

    entity_results = {}
    for entity_type in fixed_entity_types:
        df = pd.DataFrame(results[entity_type])
        # logger.info(f"Results for {entity_type}:\n{df}\n")
        df_with_mean = df.mean()
        entity_results[entity_type] = df_with_mean[['precision', 'recall', 'f1']].to_dict()

    mean_results = {'precision': 0, 'recall': 0, 'f1': 0}
    for entity_type in fixed_entity_types:
        mean_results['precision'] += entity_results[entity_type]['precision']
        mean_results['recall'] += entity_results[entity_type]['recall']
        mean_results['f1'] += entity_results[entity_type]['f1']
    mean_results['precision'] /= len(fixed_entity_types)

    logger.info(f"\n-----\nICL Entity GPT:\n{mean_results}\n------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--icl_examples_path', type=str, required=True)
    parser.add_argument('--eval_path', type=str, required=True)
    parser.add_argument('--entity_types', type=str, nargs='+', required=False,
                        help='Entity types to evaluate', default=['comb-element', 'analogy-src', 'analogy-target'])
    parser.add_argument('--nr_samples_per_class', type=int, default=1)
    parser.add_argument('--nr_repeats', type=int, default=1)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--openai_engine', type=str, default='gpt-4o-mini')
    args = parser.parse_args()

    OPEN_AI_CLIENT = init_openai_client()

    logger = setup_default_logger(args.output_dir)

    main(args.icl_examples_path, args.eval_path, args.entity_types, args.nr_samples_per_class, args.nr_repeats)
