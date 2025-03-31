import argparse
import json
import re

import pandas as pd
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from tqdm import tqdm
from util import setup_default_logger


def parse_output(raw_out: str) -> str:
    re_pattern = '<classification>(.*?)</classification>'
    matches = re.findall(re_pattern, raw_out, re.DOTALL)
    if not matches:
        re_pattern = '>(.*?)<'
        matches = re.findall(re_pattern, raw_out, re.DOTALL)
        matches = ['>' + match.strip() + '<' for match in matches]
    if not matches:
        matches = [raw_out.replace(f'<classification>', '').replace(f'</classification>', '')]

    for match in matches:
        raw_out = match.strip()
        if raw_out in ['relevant', 'irrelevant']:
            return raw_out
    return ''


def compute_metrics(predicted_classes, reference_classes, classes):
    metrics = {}
    for pos_class in classes:
        tp = sum([1 for pred, ref in zip(predicted_classes, reference_classes) if
                  pred == pos_class and ref == pos_class])
        fp = sum([1 for pred, ref in zip(predicted_classes, reference_classes) if
                  pred == pos_class and ref != pos_class])
        fn = sum([1 for pred, ref in zip(predicted_classes, reference_classes) if
                  pred != pos_class and ref == pos_class])

        precision = tp / (tp + fp) if tp + fp > 0 else -1
        recall = tp / (tp + fn) if tp + fn > 0 else -1
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else -1
        metrics.update({f"{pos_class}_f1": f1, f"{pos_class}_precision": precision, f"{pos_class}_recall": recall})

    return metrics


def main(eval_path: str, tokenizer_path: str, model_path: str, lora_path: str):
    eval_examples = pd.read_json(eval_path, lines=True)
    logger.info(f"Loaded {len(eval_examples)} test examples from {eval_path}")

    tokenizer = MistralTokenizer.from_file(tokenizer_path)
    logger.info(f"Loaded tokenizer from {tokenizer_path}")

    model = Transformer.from_folder(model_path)
    logger.info(f"Loaded model from {model_path}")

    model.load_lora(lora_path)
    logger.info(f"Loaded Lora from {lora_path}")

    pbar = tqdm(total=len(eval_examples), desc='Generating completions...')
    results = []
    pred_classes = []
    gold_classes = []

    for idx, example in eval_examples.iterrows():
        messages = example['messages']
        assert len(messages) > 0 and 'role' in messages[0] and messages[0]['role'] == 'user'
        assert 'content' in messages[0]

        user_prompt = messages[0]['content']
        completion_request = ChatCompletionRequest(messages=[UserMessage(content=user_prompt)])
        tokens = tokenizer.encode_chat_completion(completion_request).tokens
        out_tokens, _ = generate([tokens], model, max_tokens=150, temperature=0.0,
                                 eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
        raw_result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
        result = parse_output(raw_result)

        assert 'role' in messages[1] and messages[1]['role'] == 'assistant'
        assert 'content' in messages[1]
        gold_output_start = messages[1]['content'].find('<classification>') + len('<classification>')
        gold_output_end = messages[1]['content'].find('</classification>')
        gold_output = messages[1]['content'][gold_output_start:gold_output_end].strip()
        assert gold_output in ['relevant', 'irrelevant']

        logger.info('----------------------')
        logger.info(f"#User Prompt#\n{user_prompt}\n")
        logger.info(f"#Raw Output#\n{raw_result}\n")
        logger.info(f"#Output: {result} || Gold Output: {gold_output}")
        logger.info('----------------------')

        pred_classes.append(result)
        gold_classes.append(gold_output)

        results.append({'input': user_prompt, 'raw_output': raw_result, 'output': result, 'gold_output': gold_output})

        pbar.update(1)

    classes = ['relevant', 'irrelevant']
    metrics = compute_metrics(predicted_classes=pred_classes, reference_classes=gold_classes, classes=classes)

    metrics_by_category = {}
    for category in ['relevant', 'irrelevant']:
        metrics_by_category[category] = {k.split(f'{category}_')[-1]: v for k, v in metrics.items() if
                                         k.startswith(category)}

    mean_f1 = sum([metrics_by_category[category]['f1'] for category in classes]) / len(classes)
    mean_precision = sum([metrics_by_category[category]['precision'] for category in classes]) / len(classes)
    mean_recall = sum([metrics_by_category[category]['recall'] for category in classes]) / len(classes)

    mean_metrics = {'precision': mean_precision, 'recall': mean_recall, 'f1': mean_f1}
    model_name = 'Mistral-'
    if args.is_cot:
        model_name = 'COT-'
    logger.info(f"\n-----\n{model_name}-Classifier: {json.dumps(mean_metrics, indent=4)}\n-----")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_path', type=str, required=True)
    parser.add_argument('--tokenizer_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--lora_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--is-cot', action='store_true')

    args = parser.parse_args()
    logger = setup_default_logger(args.output_dir)
    main(args.eval_path, args.tokenizer_path, args.model_path, args.lora_path)
