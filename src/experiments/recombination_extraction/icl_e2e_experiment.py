import argparse
import json
import re
from typing import List, Dict

import pandas as pd
import time
from tqdm import tqdm
from util import init_openai_client, setup_default_logger, request_openai_batch_completions, \
    get_openai_batch_completions, NER_ENTITY_TYPES_ATTRIBUTES
from eval import RelationEntity, Relation, eval_rel_extraction, compute_entity_agreement

PROMPT_E2E = """You are an AI assistant tasked with analyzing scientific abstracts for idea recombination. Your goal is to identify the most salient recombination in a given abstract and format it as a JSON string. Follow these instructions carefully:

1. First, familiarize yourself with the possible entity types for recombinations:

<entity_types>
{ENTITY_TYPE_DESCRIPTIONS}
</entity_types>

2. Review the following examples to understand the expected output format and the process of identifying recombinations:

<examples>
{EXAMPLES}
</examples>

3. Now, carefully read the following scientific abstract:

<abstract>
{TEXT}
</abstract>

4. Your task is to extract the most salient recombination from this abstract. A recombination can be either:
   a) Combination: The authors combine two or more ideas, methods, models, techniques, or approaches to obtain a certain goal.
   b) Inspiration: The authors draw inspiration or similarities from one concept, idea, problem, approach, or domain and implement it in another.

5. After identifying the recombination, you will format it as a JSON string in the following structure:

   <recombination>
   {recombination_type: {entity_type_1: [ent_1, ent_2], entity_type_2: [ent_3],...}}
   </recombination>

   If you don't think the text discusses a recombination, or that the recombination is not a central part of the work, return an empty JSON object: {}.

6. Before providing your final answer, use the following scratchpad to think through the process:

   <scratchpad>
   1. Identify the main ideas, methods, or approaches discussed in the abstract.
   2. Determine if there is a clear combination of ideas or if one idea inspired the application in another domain.
   3. Identify the specific entities involved in the recombination.
   4. Classify the entities according to the provided entity types.
   5. Determine the recombination type (combination or inspiration).
   </scratchpad>

7. Now, provide your final output in the specified JSON format. Ensure that the output is a valid JSON string. If the output is empty, return {}. Place your answer within <recombination> tags.

Remember to carefully analyze the abstract and only identify a recombination if it is clearly present and central to the work described."""


def prep_prompt_template(icl_examples: pd.DataFrame, entity_types: List[str]) -> str:
    prompt_template = PROMPT_E2E
    examples_text = ''
    example_idx = 1
    for _, example in icl_examples.iterrows():
        text = example['text']
        entities = example['entities']
        rel_type = example['type']
        example_out = {rel_type: entities}
        if rel_type == 'no_recombination':
            example_out = {}
        example_text = f"Example {example_idx}:\n<input>\n{text}\n</input>\n<output>\n{json.dumps(example_out)}\n<output>\n\n"
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


def extract_json_output(raw_out: str, pattern: str) -> Dict:
    re_pattern = f'<{pattern}>(.*?)</{pattern}>'
    matches = re.findall(re_pattern, raw_out, re.DOTALL)
    parsed_answer = {}
    if not matches:
        raw_out = raw_out.replace(f'</{pattern}>', '')
        raw_out = raw_out.replace(f'<{pattern}>', '')
        matches = [raw_out]
    for match in matches:
        out = match.strip()
        try:
            parsed_answer = json.loads(out)
        except json.decoder.JSONDecodeError:
            logger.info(f"Failed to parse JSON from:\nmatch:\n{out}\n-----")
            parsed_answer = {}
            continue

        if isinstance(parsed_answer, str):
            parsed_answer = json.loads(parsed_answer)

        return parsed_answer

    return parsed_answer


def extract_recomb_from_relations(readable_relations: Dict) -> Dict:
    if 'analogy' in readable_relations:
        inspiration_entities = readable_relations['analogy'][0]
        inspiration_entities['inspiration-src'] = inspiration_entities.pop('analogy-src')
        inspiration_entities['inspiration-target'] = inspiration_entities.pop('analogy-target')
        return {'type': 'inspiration', 'entities': inspiration_entities}
    elif 'combination' in readable_relations:
        recombination_entities = readable_relations['combination']
        return {'type': 'combination', 'entities': recombination_entities[0]}

    return {}


def get_icl_set(icl_examples: pd.DataFrame, nr_samples_per_class: int):
    recombination_examples = icl_examples[icl_examples['document_class'] == 'relevant']
    inspiration_examples = []
    combination_examples = []
    for _, example in recombination_examples.iterrows():
        recombination = json.loads(example['readable_relations'])
        if 'analogy' in recombination:
            inspiration_entities = recombination['analogy'][0]
            inspiration_entities['inspiration-src'] = inspiration_entities.pop('analogy-src')
            inspiration_entities['inspiration-target'] = inspiration_entities.pop('analogy-target')
            inspiration_examples.append(
                {'text': example['text'], 'type': 'inspiration', 'entities': inspiration_entities})
        elif 'combination' in recombination:
            recombination_entities = recombination['combination']
            combination_examples.append(
                {'text': example['text'], 'type': 'combination', 'entities': recombination_entities[0]})

    inspiration_examples = pd.DataFrame(inspiration_examples)
    combination_examples = pd.DataFrame(combination_examples)

    irrelevant_examples = icl_examples[icl_examples['document_class'] == 'irrelevant']
    no_recombination_examples = []
    for _, example in irrelevant_examples.iterrows():
        no_recombination_examples.append({'text': example['text'], 'entities': {}, 'type': 'no_recombination'})

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


def average_results(results: List[Dict]) -> Dict:
    avg_results = {}
    for repeat_results in results:
        for key, value in repeat_results.items():
            if key not in avg_results:
                avg_results[key] = {}
            for sub_key, sub_value in value.items():
                if sub_key not in avg_results[key]:
                    avg_results[key][sub_key] = []
                avg_results[key][sub_key].append(sub_value)

    for key, value in avg_results.items():
        for sub_key, sub_value in value.items():
            avg_results[key][sub_key] = sum(sub_value) / len(sub_value)

    return avg_results


def main(icl_examples_path: str, eval_examples_path: str, entity_types: List[str], nr_samples_per_class: int,
         nr_repeats: int):
    icl_examples, test_examples = load_data(icl_examples_path, eval_examples_path)

    repeat_batch_ids = []

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

        examples = {}
        for _, example in test_examples.iterrows():
            text = example['text']
            prompt = prompt_template.replace('{TEXT}', text)
            examples[example['paper_id']] = prompt

        batch_id = request_openai_batch_completions(examples, args.max_tokens, args.temperature, i, args.output_dir,
                                                    OPEN_AI_CLIENT, args.openai_engine)
        logger.info(f'Batch id for repeat {i + 1}: {batch_id}')
        repeat_batch_ids.append(batch_id)

    responses = {}
    while len(responses) < len(repeat_batch_ids):
        for batch_id in repeat_batch_ids:
            if batch_id not in responses:
                batch_completions, _ = get_openai_batch_completions(batch_id, OPEN_AI_CLIENT)
                if batch_completions:
                    responses[batch_id] = batch_completions
        time.sleep(60)

    responses = list(responses.values())

    results_relations = []
    results_entities = []
    results_classes = []

    for idx, repeat_responses in enumerate(responses):
        pred_rels = []
        pred_entities = []
        gold_rels = []
        gold_entities = []
        texts = []
        pbar = tqdm(total=len(test_examples), desc=f'Evaluating repeat {idx + 1}...')

        for _, example in test_examples.iterrows():
            text = example['text']
            texts.append(text)

            pred_recomb = extract_json_output(repeat_responses[example['paper_id']], 'recombination')

            extracted_relations_pred = {}
            extracted_entities_pred = []

            if pred_recomb:
                recomb_type = list(pred_recomb.keys())[0]
                recomb_entities = pred_recomb[recomb_type]
                relation = Relation.from_entity_dictionaries(recomb_type, recomb_entities, {})
                extracted_relations_pred[relation.relation_type] = [relation]
                for entity_type, entities in recomb_entities.items():
                    extracted_entities_pred.extend([RelationEntity('', entity_type, ent, []) for ent in entities])

            pred_entities.append(extracted_entities_pred)
            pred_rels.append(extracted_relations_pred)

            gold_output = json.loads(example['readable_relations'])
            gold_recomb = extract_recomb_from_relations(gold_output)

            gold_relations = {}
            doc_gold_entities = []
            if gold_recomb:
                relation = Relation.from_entity_dictionaries(gold_recomb['type'], gold_recomb['entities'], {})
                gold_relations[relation.relation_type] = [relation]
                for entity_type, entities in gold_recomb['entities'].items():
                    doc_gold_entities.extend([RelationEntity('', entity_type, ent, []) for ent in entities])
            gold_rels.append(gold_relations)
            gold_entities.append(doc_gold_entities)
            pbar.update(1)

        rel_results, _, class_results = eval_rel_extraction(pred_rels, gold_rels, texts,
                                                            ['combination', 'inspiration'],
                                                            ['comb-element', 'inspiration-src',
                                                             'inspiration-target'],
                                                            logger, 0.6)
        entity_results = compute_entity_agreement(pred_entities, gold_entities, texts,
                                                  ['comb-element', 'inspiration-src',
                                                   'inspiration-target'],
                                                  logger)

        # logger.info(f'class_results repeat {idx + 1}: {class_results}')
        # logger.info(f'entity_results repeat {idx + 1}: {entity_results}')
        # logger.info(f'rel_results repeat {idx + 1}: {rel_results}')

        results_classes.append(class_results)
        results_entities.append(entity_results)
        results_relations.append(rel_results)

    avg_results_entities = average_results(results_entities)['avg']
    avg_results_entities = {'precision': avg_results_entities['precision'], 'recall': avg_results_entities['recall'],
                            'f1': avg_results_entities['f1']}
    avg_results_relations = average_results(results_relations)['avg']
    avg_results_relations = {'precision': avg_results_relations['precision'], 'recall': avg_results_relations['recall'],
                             'f1': avg_results_relations['f1']}
    avg_results_classes = average_results(results_classes)['avg']
    avg_results_classes = {'precision': avg_results_classes['precision'], 'recall': avg_results_classes['recall'],
                           'f1': avg_results_classes['f1']}

    all_results = {'relation-extraction': avg_results_relations, 'entity-extraction': avg_results_entities,
                   'classification': avg_results_classes}

    logger.info(f'\n-----\nICL E2E GPT: {json.dumps(all_results, indent=2)}\n-----\n')


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
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--max_tokens', type=int, default=2048)
    args = parser.parse_args()

    OPEN_AI_CLIENT = init_openai_client()

    logger = setup_default_logger(args.output_dir)

    main(args.icl_examples_path, args.eval_path, args.entity_types, args.nr_samples_per_class, args.nr_repeats)
