import json
import os
import re
from typing import Dict, List
import argparse

import pandas as pd
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from tqdm import tqdm
from util import setup_default_logger, NER_ENTITY_TYPES_ATTRIBUTES
from eval import eval_rel_extraction, Relation, RelationEntity, compute_entity_agreement

PROMPT_E2E = """You are an AI assistant tasked with analyzing scientific abstracts for idea recombination. Your goal is to identify the most salient recombination in the given abstract and format it as a JSON string. Follow these instructions carefully:

1. First, familiarize yourself with the possible entity types for recombinations:

<entity_types>
{ENTITY_TYPE_DESCRIPTIONS}
</entity_types>

2. Now, carefully read the following scientific abstract:

<abstract>
{TEXT}
</abstract>

3. Your task is to extract the most salient recombination from this abstract. A recombination can be either:
   a) Combination: The authors combine two or more ideas, methods, models, techniques, or approaches to obtain a certain goal.
   b) Inspiration: The authors draw inspiration or similarities from one concept, idea, problem, approach, or domain and implement it in another.

4. After identifying the recombination, you will format it as a JSON string in the following structure:

   <recombination>
   {recombination_type: {entity_type_1: [ent_1, ent_2], entity_type_2: [ent_3],...}}
   </recombination>

   If you don't think the text discusses a recombination, or that the recombination is not a central part of the work, return an empty JSON object: {}.

5. Before providing your final answer, use the following scratchpad to think through the process:

   <scratchpad>
   1. Identify the main ideas, methods, or approaches discussed in the abstract.
   2. Determine if there is a clear combination of ideas or if one idea inspired the application in another domain.
   3. Identify the specific entities involved in the recombination.
   4. Classify the entities according to the provided entity types.
   5. Determine the recombination type (combination or inspiration).
   </scratchpad>

6. Now, provide your final output in the specified JSON format. Ensure that the output is a valid JSON string. If the output is empty, return {}. Place your answer within <answer> tags.

Remember to carefully analyze the abstract and only identify a recombination if it is clearly present and central to the work described."""


def prompt_model(model: Transformer, tokenizer: MistralTokenizer, user_prompt: str) -> str:
    completion_request = ChatCompletionRequest(messages=[UserMessage(content=user_prompt)])
    tokens = tokenizer.encode_chat_completion(completion_request).tokens
    out_tokens, _ = generate([tokens], model, max_tokens=1000, temperature=0.0,
                             eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
    raw_result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
    return raw_result


def load_tokenizer(trained_model_path: str) -> MistralTokenizer:
    tokenizer_path = os.path.join(trained_model_path, 'tokenizer.model.v3')
    tokenizer = MistralTokenizer.from_file(tokenizer_path)
    logger.info(f"Loaded tokenizer from {tokenizer_path}")
    return tokenizer


def load_trained_model(base_model_path: str, trained_model_path: str) -> Transformer:
    model = Transformer.from_folder(base_model_path)
    logger.info(f"Loaded model from {base_model_path}")

    lora_path = os.path.join(trained_model_path, 'lora.safetensors')
    model.load_lora(lora_path)
    logger.info(f"Loaded Lora from {lora_path}")

    return model


def get_user_prompt(messages: List[Dict]) -> str:
    assert len(messages) > 0 and 'role' in messages[0] and messages[0]['role'] == 'user'
    assert 'content' in messages[0]
    return messages[0]['content']


def extract_json_output(raw_out: str, pattern: str) -> Dict:
    re_pattern = f'<{pattern}>(.*?)</{pattern}>'
    matches = re.findall(re_pattern, raw_out, re.DOTALL)
    if not matches:
        raw_out = raw_out.replace(f'</{pattern}>', '')
        raw_out = raw_out.replace(f'<{pattern}>', '')
        matches = [raw_out]
    for match in matches:
        out = match.strip()
        try:
            parsed_answer = json.loads(out)
        except json.decoder.JSONDecodeError:
            logger.info(f"Failed to parse JSON from:\n\nraw_out:\n{raw_out}\n\n----\n\nmatch:\n{out}\n-----")
            return {}

        if isinstance(parsed_answer, str):
            parsed_answer = json.loads(parsed_answer)

        return parsed_answer


def parse_answer(answer_text: str, entity_types: List[str]):
    recombination = extract_json_output(answer_text, 'recombination')
    if recombination:
        recombination_type = list(recombination.keys())[0]
        recombination_entities = recombination[recombination_type]
        verified_entities = {}
        for entity_type in recombination_entities:
            if entity_type in entity_types:
                verified_entities[entity_type] = recombination_entities[entity_type]

        recombination = {'type': recombination_type, 'entities': verified_entities}

    if recombination:
        if recombination['type'] == 'combination':
            comb_elements = recombination['entities'].get('comb-element', [])
            if len(comb_elements) < 2:
                recombination = None
                return recombination

        elif recombination['type'] == 'inspiration':
            inspiration_src = recombination['entities'].get('inspiration-src', [])
            inspiration_target = recombination['entities'].get('inspiration-target', [])
            if len(inspiration_src) == 0 or len(inspiration_target) == 0:
                recombination = None
                return recombination
        else:
            logger.info(f"Unknown recombination type: {recombination['type']}")
            recombination = None
            return recombination

    return recombination


def prep_assistant_answer_e2e(entities: List, relations: List):
    entities_by_id = {ent['tagged_token_id']: ent for ent in entities}
    for entity in entities_by_id.values():
        if entity['tag'] == 'analogy-src':
            entity['tag'] = 'inspiration-src'
        if entity['tag'] == 'analogy-target':
            entity['tag'] = 'inspiration-target'

    relations_by_type = {}
    for rel in relations:
        rel_type = rel['type']
        if rel_type == 'analogy':
            rel_type = 'inspiration'
        if rel_type not in relations_by_type:
            relations_by_type[rel_type] = []
        relations_by_type[rel_type].append(rel['entities'])

    assistant_answer_content = ""
    recombination = {}
    recombination_entities = []
    if 'combination' in relations_by_type:
        entities = {}
        for entity in relations_by_type['combination'][0]:
            entity_type = entities_by_id[entity]['tag']
            if entity_type not in entities:
                entities[entity_type] = []
            entities[entity_type].append(entities_by_id[entity]['value'])
            recombination_entities.append(entities_by_id[entity])
        recombination = {'combination': entities}
    elif 'inspiration' in relations_by_type:
        entities = {}
        for entity in relations_by_type['inspiration'][0]:
            entity_type = entities_by_id[entity]['tag']
            if entity_type not in entities:
                entities[entity_type] = []
            entities[entity_type].append(entities_by_id[entity]['value'])
            recombination_entities.append(entities_by_id[entity])
        recombination = {'inspiration': entities}

    assistant_answer_content += f'<recombination>\n{json.dumps(recombination, indent=4)}\n</recombination>\n'

    assistant_answer = {'role': 'assistant', 'content': assistant_answer_content}

    return assistant_answer


def process_data_e2e(test_data: pd.DataFrame):
    entity_types_description = ""
    for i, type_attributes in enumerate(NER_ENTITY_TYPES_ATTRIBUTES):
        if type_attributes['entity_type'] not in ["comb-element", "analogy-src", "analogy-target"]:
            continue

        entity_type_name = type_attributes['entity_type']
        if 'prompt_type_name' in type_attributes:
            entity_type_name = type_attributes['prompt_type_name']

        if type_attributes['entity_type'] in ["comb-element", "analogy-src", "analogy-target"]:
            entity_types_description += f"{i + 1}. {entity_type_name}: {type_attributes['desc']}\n"

    prompt_template = PROMPT_E2E.replace("{ENTITY_TYPE_DESCRIPTIONS}", entity_types_description)

    processed_examples = []
    for i, row in test_data.iterrows():
        text = row['text']
        prompt = prompt_template.replace("{TEXT}", text)
        user_msg = {'role': 'user', 'content': prompt}
        logger.info(f"User message: {user_msg['content']}")
        assistant_answer = prep_assistant_answer_e2e(json.loads(row['entities']), json.loads(row['relations']))
        logger.info(f"assistant answer: {assistant_answer['content']}")
        processed_examples.append({'messages': [user_msg, assistant_answer], 'paper_id': row['paper_id']})

    return pd.DataFrame(processed_examples)


def main(eval_path: str, base_model_path: str, trained_model_path: str):
    test_examples = pd.read_csv(eval_path, dtype={'paper_id': str})
    logger.info(f"Loaded {len(test_examples)} test examples from {eval_path}")

    test_examples = process_data_e2e(test_examples)
    entity_types = [et.replace('analogy-src', 'inspiration-src').replace('analogy-target', 'inspiration-target') for et
                    in ["comb-element", "analogy-src", "analogy-target"]]
    e2e_extraction_entity_types = entity_types.copy()

    e2e_tokenizer = load_tokenizer(trained_model_path)
    e2e_model = load_trained_model(base_model_path, trained_model_path)

    pred_rels = []
    pred_entities = []
    gold_rels = []
    gold_entities = []
    texts = []
    pbar = tqdm(total=len(test_examples), desc='Running e2e model on test data...')

    for _, example in test_examples.iterrows():
        user_prompt = get_user_prompt(example['messages'])
        raw_result = prompt_model(e2e_model, e2e_tokenizer, user_prompt)
        pred_recomb = parse_answer(raw_result, e2e_extraction_entity_types)
        text_sep = 'abstract'
        text_start = user_prompt.find(f'<{text_sep}>') + len(f'<{text_sep}>')
        text_end = user_prompt.find(f'</{text_sep}>')
        if text_start == -1 or text_end == -1:
            raise ValueError(f"Failed to find text or abstract in user prompt: {user_prompt}")
        text = user_prompt[text_start:text_end].strip()
        texts.append(text)

        gold_recomb = parse_answer(example['messages'][1]['content'], entity_types)
        extracted_relations_pred = {}
        extracted_entities_pred = []
        if pred_recomb:
            relation = Relation.from_entity_dictionaries(pred_recomb['type'], pred_recomb['entities'], {})
            extracted_relations_pred[pred_recomb['type']] = [relation]
            for entity_type, entities in pred_recomb['entities'].items():
                extracted_entities_pred.extend([RelationEntity('', entity_type, ent, []) for ent in entities])

        pred_entities.append(extracted_entities_pred)
        pred_rels.append(extracted_relations_pred)

        gold_relations = {}
        doc_gold_entities = []
        if gold_recomb:
            relation = Relation.from_entity_dictionaries(gold_recomb['type'], gold_recomb['entities'], {})
            gold_relations[gold_recomb['type']] = [relation]
            for entity_type, entities in gold_recomb['entities'].items():
                doc_gold_entities.extend([RelationEntity('', entity_type, ent, []) for ent in entities])
        gold_rels.append(gold_relations)
        gold_entities.append(doc_gold_entities)
        pbar.update(1)

    rel_results, _, class_results = eval_rel_extraction(pred_rels, gold_rels, texts,
                                                        ['combination', 'inspiration'],
                                                        entity_types,
                                                        logger, 0.6)

    print(rel_results)

    rel_results = {"precision": rel_results['avg']['precision'], "recall": rel_results['avg']['recall'],
                   "f1": rel_results['avg']['f1']}
    class_results = {"precision": class_results['avg']['precision'], "recall": class_results['avg']['recall'],
                     "f1": class_results['avg']['f1']}

    entity_results = compute_entity_agreement(pred_entities, gold_entities, texts,
                                              ['comb-element', 'inspiration-src', 'inspiration-target'],
                                              logger)

    entity_results = {"precision": entity_results['avg']['precision'], "recall": entity_results['avg']['recall'],
                      "f1": entity_results['avg']['f1']}

    total_results = {'relation_extraction': rel_results, 'entity_extraction': entity_results,
                     'classification': class_results}

    table_string = '\n-----\nMistral E2E:\nRes-type\t&\tP\t&\tR\t&\tF1 \\\\ \n'
    for res_type, res in total_results.items():
        table_string += f"{res_type[:8]}\t&\t{res['precision']:.3f}\t&\t{res['recall']:.3f}\t&\t{res['f1']:.3f} \\\\ \n"

    table_string += '\n-----\n'
    logger.info(table_string)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_path', type=str, required=True, help='Path to the evaluation data')
    parser.add_argument('--base_model_path', type=str, required=True, help='Path to the base model')
    parser.add_argument('--trained_model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory')
    args = parser.parse_args()
    logger = setup_default_logger(args.output_dir)
    main(args.eval_path, args.base_model_path, args.trained_model_path)
