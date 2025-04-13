import argparse
import json
import os
from typing import Dict

import re
from util import setup_default_logger, NER_ENTITY_TYPES_ATTRIBUTES, PROMPT_E2E
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest


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


def extract_json_output(raw_out: str, pattern: str) -> Dict:
    pattern = f'<{pattern}>(.*?)</{pattern}>'
    matches = re.findall(pattern, raw_out, re.DOTALL)
    if not matches:
        return {}
    for match in matches:
        out = match.strip()
        try:
            parsed_answer = json.loads(out)
        except json.decoder.JSONDecodeError:
            logger.info(f"Failed to parse JSON from:\n\nraw_out:\n{raw_out}\n\n----\n\nmatch:\n{out}\n-----")
            return {}

        if isinstance(parsed_answer, str):
            parsed_answer = json.loads(parsed_answer)

        if not isinstance(parsed_answer, dict):
            logger.info(f"Invalid JSON output:\n{parsed_answer}")
            return {}

        return parsed_answer


def parse_answer(answer_text: str, entity_types=['comb-element', 'inspiration-src', 'inspiration-target']):
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


def get_extraction_prompt(text: str, entity_types=['comb-element', 'analogy-src', 'analogy-target']):
    entity_types_description = ""
    for i, type_attributes in enumerate(NER_ENTITY_TYPES_ATTRIBUTES):
        if type_attributes['entity_type'] not in entity_types:
            continue
        if type_attributes['entity_type'] == 'cite':
            continue

        entity_type_name = type_attributes['entity_type']
        if 'prompt_type_name' in type_attributes:
            entity_type_name = type_attributes['prompt_type_name']

        if type_attributes['entity_type'] in entity_types:
            entity_types_description += f"{i + 1}. {entity_type_name}: {type_attributes['desc']}\n"

    prompt_template = PROMPT_E2E.replace("{ENTITY_TYPE_DESCRIPTIONS}", entity_types_description)
    prompt = prompt_template.replace("{TEXT}", text)
    return prompt


def main():
    tokenizer = load_tokenizer(args.extraction_model_path)
    model = load_trained_model(args.base_model_path, args.extraction_model_path)
    input_text = open(args.input_path).read()
    res_text = f'\n---\nINPUT:\n{input_text}\n---'

    prompt = get_extraction_prompt(input_text)
    result = prompt_model(model, tokenizer, prompt)
    recombination = parse_answer(result)

    if recombination:
        res_text += f"\n----\nRecombination extracted:\n{json.dumps(recombination, indent=2)}\n----"
    else:
        res_text += "\n----\nNo recombination extracted\n----"

    logger.info(res_text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--base_model_path', type=str, required=True)
    parser.add_argument('--extraction_model_path', type=str, required=True)

    args = parser.parse_args()
    logger = setup_default_logger(args.output_dir)
    main()
