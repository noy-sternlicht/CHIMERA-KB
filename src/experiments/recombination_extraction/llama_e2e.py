import argparse
import json
import os
from datetime import datetime
from typing import Dict, List
import re
import pandas as pd
import torch
import transformers
from datasets import Dataset
from huggingface_hub import login
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
from trl import SFTTrainer
from util import setup_default_logger, NER_ENTITY_TYPES_ATTRIBUTES
from eval import Relation, RelationEntity, compute_entity_agreement, eval_rel_extraction

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

6. Now, provide your final output in the specified JSON format. Ensure that the output is a valid JSON string. If the output is empty, return {}. Place your answer within <recombination> tags.

Remember to carefully analyze the abstract and only identify a recombination if it is clearly present and central to the work described."""


class AnnotationModelHF:
    def __init__(self, model_name: str, model_conf: Dict, access_token: str, checkpoint_path=''):
        self.access_token = access_token
        login(token=access_token)
        self.model_name = model_name
        self.temperature = model_conf['temperature']
        self.max_seq_length = model_conf['max_seq_length']
        self.best_checkpoint_path = checkpoint_path
        self.best_model = None

        self.tokenizer = self.init_tokenizer()
        self.run_name = f"{self.model_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        self.out_dir = os.path.join(model_conf['out_dir'], self.run_name)
        self.data_collator = transformers.DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        os.makedirs(self.out_dir)

    def get_model_init_args(self, use_cache: bool) -> Dict:
        model_init_args = {"device_map": {"": 0},
                           "use_cache": use_cache,
                           "pad_token_id": self.tokenizer.pad_token_id,
                           }

        if self.access_token:
            model_init_args['token'] = self.access_token

        return model_init_args

    def init_tokenizer(self) -> AutoTokenizer:
        tokenizer_init_args = {"trust_remote_code": True, "add_eos_token": True, "padding_side": "left",
                               "model_max_length": self.max_seq_length}
        if self.access_token:
            tokenizer_init_args['token'] = self.access_token

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, **tokenizer_init_args)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
        return tokenizer

    def init_trainer(self, train_dataset: Dataset) -> SFTTrainer:

        trainer_args = transformers.TrainingArguments(
            output_dir=self.out_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_checkpointing=True,
            gradient_accumulation_steps=4,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="paged_adamw_8bit",
            warmup_steps=500,
            max_steps=args.nr_steps,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            logging_steps=5,
            bf16=True,
            logging_dir="./logs",
            run_name="test_run"
        )

        peft_params = LoraConfig(
            r=64,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )

        model_args = self.get_model_init_args(use_cache=False)
        model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_args)

        trainer = SFTTrainer(
            model=model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            peft_config=peft_params,
            dataset_text_field="prompt",
            max_seq_length=self.max_seq_length,
            args=trainer_args,
            packing=False,
            data_collator=self.data_collator,
        )

        return trainer

    def generate_prompt(self, instructions: str, in_text: str, out_text: str = None):
        prompt = instructions.replace('{TEXT}', in_text)

        entity_types_description = ""
        entity_type_examples = {}
        example_idx = 1
        for i, type_attributes in enumerate(NER_ENTITY_TYPES_ATTRIBUTES):
            if type_attributes['entity_type'] not in args.entity_types:
                continue
            entity_type_name = type_attributes['entity_type']
            if 'prompt_type_name' in type_attributes:
                entity_type_name = type_attributes['prompt_type_name']

            if type_attributes['entity_type'] in args.entity_types:
                entity_types_description += f"{i + 1}. {entity_type_name}: {type_attributes['desc']}\n"

            entity_type_examples[entity_type_name] = []
            for _ in range(type_attributes['nr_entities_in_example']):
                entity_type_examples[entity_type_name].append(f"entity{example_idx}")
                example_idx += 1

        entity_types_description = entity_types_description.strip()
        prompt = prompt.replace('{ENTITY_TYPE_DESCRIPTIONS}', entity_types_description)
        prompt = prompt.replace('{ENTITY_TYPE_EXAMPLES}', str(entity_type_examples))

        messages = [{"role": "user", "content": prompt}]

        if out_text:
            out_dict = json.loads(out_text)
            if out_dict:
                recomb_type = list(out_dict.keys())[0]
                recomb_entities = out_dict[recomb_type][0]
                if recomb_type == 'analogy':
                    recomb_type = 'inspiration'
                if 'analogy-src' in recomb_entities:
                    recomb_entities['inspiration-src'] = recomb_entities.pop('analogy-src')
                if 'analogy-target' in recomb_entities:
                    recomb_entities['inspiration-target'] = recomb_entities.pop('analogy-target')
                recomb_dict = {recomb_type: recomb_entities}
                out_text = f'<recombination>\n{json.dumps(recomb_dict)}\n</recombination>'
            else:
                out_text = '<recombination>\n{}\n</recombination>'
            messages.append({"role": "assistant", "content": out_text})
            result = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        else:
            result = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return result

    def create_dataset(self, data: pd.DataFrame, input_field: str, output_field: str, prompt_template: str,
                       for_eval=False) -> Dataset:
        data_rows = []

        for _, row in data.iterrows():
            if for_eval:
                prompt = self.generate_prompt(prompt_template, row[input_field])
            else:
                prompt = self.generate_prompt(prompt_template, row[input_field], row[output_field])
            data_rows.append(
                {
                    'paper_id': row['paper_id'],
                    'prompt': prompt,
                    'input': row[input_field],
                    'output': row[output_field]
                }
            )
        data_df = pd.DataFrame(data_rows)
        generated_dataset = Dataset.from_pandas(data_df)
        return generated_dataset

    def finetune_model(self, training_data: pd.DataFrame, input_field: str, output_field: str, prompt_template: str):
        logger.info(f"Finetuning model {self.model_name} on {len(training_data)} examples.")
        train_dataset = self.create_dataset(training_data, input_field, output_field, prompt_template)
        logger.info(f"Created training dataset with {len(train_dataset)} examples.")

        trainer = self.init_trainer(train_dataset)
        trainer.train()
        self.best_checkpoint_path = os.path.join(self.out_dir, f'best_checkpoint_{self.run_name}')
        trainer.save_model(self.best_checkpoint_path)
        self.best_model = trainer.model
        logger.info(f'Saved best checkpoint to: {self.best_checkpoint_path}')

        return self.best_checkpoint_path

    def prompt_model(self, model: AutoModelForCausalLM, prompt: str) -> str:
        model.eval()
        model_inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**model_inputs,
                                     max_length=self.max_seq_length, return_dict_in_generate=True,
                                     output_scores=True, pad_token_id=self.tokenizer.pad_token_id)

        input_length = model_inputs['input_ids'].shape[1]
        out_tokens = outputs.sequences[0][input_length:]
        out = self.tokenizer.decode(out_tokens, skip_special_tokens=True)

        return out

    def test_model(self, test_data: pd.DataFrame, input_field: str, output_field: str, prompt_template: str,
                   results_file_prefix: str, checkpoint_path=None) -> pd.DataFrame:
        model_args = self.get_model_init_args(use_cache=True)
        if checkpoint_path:
            self.best_checkpoint_path = checkpoint_path

        model = AutoModelForCausalLM.from_pretrained(self.best_checkpoint_path, **model_args)
        self.tokenizer.padding_side = "left"

        results = []
        pbar = tqdm(total=len(test_data), desc='Generating predictions...')
        for row_id, row in test_data.iterrows():
            prompt = self.generate_prompt(prompt_template, row[input_field])
            result = {
                'paper_id': row['paper_id'],
                input_field: row[input_field],
                'prompt': prompt,
                'raw_out': "",
            }

            raw_out = self.prompt_model(model, prompt)
            result['raw_out'] = raw_out
            results.append(result)
            pbar.update(1)

        results_df = pd.DataFrame(results)
        results_csv_path = os.path.join(self.out_dir, f'{results_file_prefix}.csv')
        results_df.to_csv(results_csv_path)

        readable_results_path = os.path.join(self.out_dir, f'{results_file_prefix}.txt')
        with open(readable_results_path, 'w') as f:
            f.write('PREDICTED\n\n')
            for res in results:
                f.write(f'====={res["paper_id"]}=====\n')
                f.write(
                    f'PROMPT:\n{res["prompt"]}\n'
                    f'OUTPUT:\n{res["raw_out"]}\n'
                )
        return results_df


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


def parse_output(answer_text: str, entity_types: List[str]):
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


def main():
    model_conf = {
        'temperature': args.temperature,
        'max_seq_length': args.max_seq_length,
        'out_dir': args.output_dir
    }

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    hf_token = open(args.hf_key_path, 'r').read().strip()
    model = AnnotationModelHF(args.model_name, model_conf, hf_token)

    logger.info(f'Reading data from {args.train_path} and {args.eval_path}')
    train_data = pd.read_csv(args.train_path, dtype={'paper_id': str})
    eval_data = pd.read_csv(args.eval_path, dtype={'paper_id': str})

    data_splits = {'train': train_data, 'eval': eval_data}

    for split_name, split_data in data_splits.items():
        logger.info(f"Loaded {len(data_splits[split_name])} examples for {split_name} split.")

    best_checkpoint_path = args.checkpoint_path
    if not best_checkpoint_path:
        best_checkpoint_path = model.finetune_model(data_splits['train'], 'text', 'readable_relations', PROMPT_E2E)
    results = model.test_model(data_splits['eval'], 'text', 'readable_relations', PROMPT_E2E, 'eval_results',
                               best_checkpoint_path)

    pred_rels = []
    pred_entities = []
    gold_rels = []
    gold_entities = []
    texts = []

    pbar = tqdm(total=len(results), desc='Processing results...')
    for _, row in data_splits['eval'].iterrows():
        row_results = results[results['paper_id'] == row['paper_id']].iloc[0]
        raw_result = row_results['raw_out']
        pred_recomb = parse_output(raw_result, ['comb-element', 'inspiration-src', 'inspiration-target'])

        text = row['text']
        texts.append(text)

        gold_recomb = json.loads(row['readable_relations'])
        if gold_recomb:
            rel_type = list(gold_recomb.keys())[0]
            rel_entities = gold_recomb[rel_type][0]
            if rel_type == 'analogy':
                rel_type = 'inspiration'

            if 'analogy-src' in rel_entities:
                rel_entities['inspiration-src'] = rel_entities.pop('analogy-src')
            if 'analogy-target' in rel_entities:
                rel_entities['inspiration-target'] = rel_entities.pop('analogy-target')
            gold_recomb = {rel_type: rel_entities}
        else:
            gold_recomb = {}

        gold_recomb = json.dumps(gold_recomb)
        gold_recomb = parse_output(gold_recomb, ['comb-element', 'inspiration-src', 'inspiration-target'])

        extracted_relations_pred = {}
        extracted_entities_pred = []
        if pred_recomb:
            relation = Relation.from_entity_dictionaries(pred_recomb['type'], pred_recomb['entities'], {})
            extracted_relations_pred[relation.relation_type] = [relation]
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
                                                        ['comb-element', 'inspiration-src', 'inspiration-target'],
                                                        logger, 0.6)

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


    logger.info(f"\n-----\nICL E2E GPT: {json.dumps(total_results, indent=4)}\n-----")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--eval_path', type=str)
    parser.add_argument('--nr_steps', type=int)
    parser.add_argument('--temperature', type=float)
    parser.add_argument('--max_seq_length', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--hf_key_path', type=str)
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--entity_types', default=["comb-element",
                                                   "analogy-src",
                                                   "analogy-target"], type=str, nargs='+')

    args = parser.parse_args()
    logger = setup_default_logger(args.output_dir)

    main()
