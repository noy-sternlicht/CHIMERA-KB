import argparse
import json
import os
import uuid
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np
from huggingface_hub import login
from scipy.special import softmax

import pandas as pd
import nltk
import transformers
from datasets import Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, EvalPrediction
from util import word_tokenize_text, map_chars_into_words, setup_default_logger

from eval import RelationEntity, compute_entity_agreement

id2label = {
    0: "comb-element",
    1: "analogy-src",
    2: "analogy-target",
    3: "other"
}


def parse_predictions(predictions: np.ndarray, labels: np.ndarray) -> (
        List[List[int]], List[List[int]], List[List[float]]):
    predictions = softmax(predictions, axis=2)
    predictions_prob = np.max(predictions, axis=2)
    predictions = np.argmax(predictions, axis=2)
    predicted_labels = [
        [pred for (pred, l, prob) in zip(prediction, label, probability) if l != -100]
        for prediction, label, probability in zip(predictions, labels, predictions_prob)
    ]

    true_labels = [
        [l for (pred, l, prob) in zip(prediction, label, probability) if l != -100]
        for prediction, label, probability in zip(predictions, labels, predictions_prob)
    ]

    predicted_labels_prob = [
        [prob for (pred, l, prob) in zip(prediction, label, probability) if l != -100]
        for prediction, label, probability in zip(predictions, labels, predictions_prob)
    ]

    return predicted_labels, true_labels, predicted_labels_prob


class TokenClassifier:
    def __init__(self, model_name: str, data_splits: Dict[str, pd.DataFrame]):
        self.token = open(args.hf_token).read().strip()
        login(token=self.token)
        self.id2label = id2label
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.model_name = model_name
        self.tokenizer = self.init_tokenizer()
        self.data_collator = transformers.DataCollatorForTokenClassification(tokenizer=self.tokenizer, padding=True)
        self.tokenized_datasets = self.init_datasets(data_dfs=data_splits)

        self.run_name = f"{self.model_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        self.out_dir = os.path.join(args.output_dir, self.run_name)
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

    def init_tokenizer(self):
        tokenizer_init_args = {"trust_remote_code": True, "add_prefix_space": True}
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, **tokenizer_init_args)
        if not hasattr(tokenizer, 'model_max_length') or tokenizer.model_max_length is None:
            tokenizer.model_max_length = 512
        return tokenizer

    def get_model_init_args(self) -> Dict:

        model_init_args = {"device_map": None,
                           "pad_token_id": self.tokenizer.pad_token_id,
                           "num_labels": len(self.id2label),
                           "id2label": self.id2label,
                           "label2id": self.label2id,
                           }

        return model_init_args

    def get_tagged_words(self, entity_annotations: List[Dict], words: List[str], text: str) -> List[int]:
        char_to_word_index = map_chars_into_words(words, text)
        word_tags = [NO_ANNOTATION_LABEL] * len(words)

        for annotation in entity_annotations:
            if text[annotation['start']:annotation['end']] != annotation['value']:
                logger.error(
                    f'Original entity value: {annotation["value"]} != new location val: {text[annotation["start"]:annotation["end"]]}')
            for char_index in range(annotation['start'], annotation['end']):
                if char_index in char_to_word_index and annotation['tag'] in self.label2id:
                    word_index = char_to_word_index[char_index]
                    word_tags[word_index] = self.label2id[annotation['tag']]

        return word_tags

    def init_datasets(self, data_dfs: Dict[str, pd.DataFrame]) -> Dict[str, Dataset]:
        datasets = {}
        for split_name in data_dfs.keys():
            split_examples = []
            split_data = data_dfs[split_name]
            for _, row in split_data.iterrows():
                example = {'id': row['paper_id'], 'text': row['text'], 'words': word_tokenize_text(row['text']),
                           'ner_tags': None, 'gold_entities': json.loads(row['readable_entities'])}
                entities = json.loads(row['entities'])
                tagged_words = self.get_tagged_words(entities, words=example['words'], text=row['text'])

                example['ner_tags'] = tagged_words
                split_examples.append(example)
            datasets[split_name] = Dataset.from_list(split_examples)
            logger.info(f'Loaded {len(datasets[split_name])} examples for {split_name} split')

        return {
            split_name: datasets[split_name].map(self.tokenize_and_align_labels, batched=True)
            for split_name in data_dfs.keys()}

    def tokenize_and_align_labels(self, examples: Dict) -> Dict:
        tokenized_inputs = self.tokenizer(examples['words'], truncation=True, is_split_into_words=True)
        labels = []
        tokenizer_word_ids = []
        for i, ner_tags in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(ner_tags[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
            tokenizer_word_ids.append(word_ids)
        tokenized_inputs["labels"] = labels
        tokenized_inputs["word_ids"] = tokenizer_word_ids

        return tokenized_inputs

    def compute_metrics(self, p: EvalPrediction):
        predicted_labels, true_labels, _ = parse_predictions(p.predictions, p.label_ids)

        metrics = {}
        label_ids = list(self.label2id.values())
        for prediction, reference in zip(predicted_labels, true_labels):
            for label_id in label_ids:
                label_name = self.id2label[label_id]
                metrics[f'{label_name}_precision'] = metrics.get(f'{label_name}_precision', [])
                metrics[f'{label_name}_recall'] = metrics.get(f'{label_name}_recall', [])
                tp, fp, fn = 0, 0, 0
                for pred, ref in zip(prediction, reference):
                    if ref == label_id:
                        if pred == label_id:
                            tp += 1
                        else:
                            fn += 1
                    elif pred == label_id:
                        fp += 1
                if tp + fp > 0:
                    metrics[f'{label_name}_precision'].append(tp / (tp + fp))
                if tp + fn > 0:
                    metrics[f'{label_name}_recall'].append(tp / (tp + fn))

        for label_id in label_ids:
            label_name = self.id2label[label_id]
            nr_elements_with_precision = len(metrics[f'{label_name}_precision'])
            metrics[f'{label_name}_precision'] = sum(metrics[
                                                         f'{label_name}_precision']) / nr_elements_with_precision if nr_elements_with_precision > 0 else -1
            nr_elements_with_recall = len(metrics[f'{label_name}_recall'])
            metrics[f'{label_name}_recall'] = sum(
                metrics[f'{label_name}_recall']) / nr_elements_with_recall if nr_elements_with_recall > 0 else -1

        return metrics

    def init_trainer(self, train_dataset: Dataset, model_init, inference_mode: bool) -> transformers.Trainer:

        training_args = transformers.TrainingArguments(output_dir=self.out_dir,
                                                       gradient_checkpointing=True,
                                                       gradient_accumulation_steps=4,
                                                       gradient_checkpointing_kwargs={"use_reentrant": False},
                                                       optim="paged_adamw_8bit",
                                                       per_device_train_batch_size=1,
                                                       max_steps=500,
                                                       weight_decay=0.1,
                                                       lr_scheduler_type="cosine",
                                                       learning_rate=6.e-5,
                                                       logging_steps=10,
                                                       run_name="test",
                                                       warmup_ratio=0.1
                                                       )

        model_init_func = model_init
        initialized_model = model_init_func()

        trainer = transformers.Trainer(
            model=initialized_model,
            train_dataset=train_dataset,
            args=training_args,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )

        return trainer

    def finetune_model(self) -> Tuple[str, Dict[str, Dict]]:
        def init_model():
            model_args = self.get_model_init_args()
            return AutoModelForTokenClassification.from_pretrained(self.model_name, **model_args)

        trainer = self.init_trainer(self.tokenized_datasets['train'], init_model, inference_mode=False)
        trainer.train()
        best_checkpoint_path = os.path.join(self.out_dir, f'best_checkpoint_{self.run_name}')
        trainer.save_model(best_checkpoint_path)
        logger.info(f'Saved best checkpoint to: {best_checkpoint_path}')

        metrics = self.eval_model(trainer, ['test'])
        return best_checkpoint_path, metrics

    def parse_results(self, raw_results: List[Dict], id2label: Dict[int, str]) -> List[Dict]:
        parsed_results = []
        for example in raw_results:
            res = {
                'paper_id': example['paper_id'],
                'text': example['text'],
                'document_class': 'irrelevant',
                'entities': [],
                "relations": [],
                "gold_entities": example['gold_entities'],
                "pred_entity_objects": [],
                "gold_entity_objects": [],
            }

            word_tokens_to_chars = map_word_tokens_indices_to_char_indices(res['text'])
            predicted_ner = get_ner_annotations_from_tagged_words(example['predicted_labels'],
                                                                  example['predicted_labels_prob'], id2label,
                                                                  word_tokens_to_chars, res['text'])
            res['entities'] = predicted_ner
            readable_entities = {}
            for ent in predicted_ner:
                res['pred_entity_objects'].append(RelationEntity(ent['tagged_token_id'], ent['tag'], ent['value'], []))
                if ent['tag'] not in readable_entities:
                    readable_entities[ent['tag']] = []
                readable_entities[ent['tag']].append(ent['value'])

            res['readable_entities'] = readable_entities

            for ent_type, entities in res['gold_entities'].items():
                if entities:
                    for entity in entities:
                        res['gold_entity_objects'].append(RelationEntity(str(uuid.uuid4()), ent_type, entity, []))

            parsed_results.append(res)

        return parsed_results

    def eval_model(self, trainer, eval_splits) -> Dict:
        results = {split_name: [] for split_name in eval_splits}
        metrics = {}
        for split_name in eval_splits:
            predict_results = trainer.predict(self.tokenized_datasets[split_name])
            predicted_labels, true_labels, predictions_probs = parse_predictions(predict_results.predictions,
                                                                                 predict_results.label_ids)
            metrics[split_name] = predict_results.metrics
            for prediction, reference, probs, example in zip(predicted_labels, true_labels, predictions_probs,
                                                             self.tokenized_datasets[split_name]):
                example_out = {'paper_id': example['id'],
                               'text': example['text'],
                               'words': example['words'],
                               'word_ids': example['word_ids'],
                               'predicted_labels': prediction,
                               'predicted_labels_prob': probs,
                               'true_labels': reference,
                               'gold_entities': example['gold_entities']
                               }
                results[split_name].append(example_out)

            parsed_split_results = self.parse_results(results[split_name], self.id2label)
            split_res_path = os.path.join(self.out_dir, f'{split_name}_results.csv')
            split_results_df = pd.DataFrame(parsed_split_results)
            split_results_df.to_csv(split_res_path)
            logger.info(f'Saved {split_name} results to: {split_res_path}')

            readable_results = ''
            docs_pred_ents = []
            pred_ent_objects = []
            docs_gold_ents = []
            docs_gold_ents_objects = []
            texts = []
            for res in parsed_split_results:
                readable_results += f'---{res["paper_id"]}---\n'
                readable_results += res['text'] + '\n'
                readable_results += f'Predicted Entities: {res["readable_entities"]}\n\n'
                readable_results += f'True Entities: {res["gold_entities"]}\n\n'
                docs_pred_ents.append(res['entities'])
                pred_ent_objects.append(res['pred_entity_objects'])
                docs_gold_ents.append(res['gold_entities'])
                docs_gold_ents_objects.append(res['gold_entity_objects'])
                texts.append(res['text'])

                logger.info(
                    f'------\nPredicted Entities: {res["readable_entities"]}\nTrue Entities: {res["gold_entities"]}\n------\n')

            entity_results = compute_entity_agreement(pred_ent_objects, docs_gold_ents_objects, texts,
                                                      list(self.id2label.values()), logger)

            entity_types = ['comb-element', 'analogy-src', 'analogy-target']

            mean_results = {'precision': 0, 'recall': 0, 'f1': 0}
            for entity_type, results in entity_results.items():
                if entity_type not in entity_types:
                    continue
                mean_results['precision'] += results['precision']
                mean_results['recall'] += results['recall']
                mean_results['f1'] += results['f1']
            mean_results['precision'] /= len(entity_types)
            mean_results['recall'] /= len(entity_types)
            mean_results['f1'] /= len(entity_types)

            logger.info(f"\n-----\nToken Classifier Results:\n{json.dumps(mean_results, indent=2)}\n------")

            readable_res_out_path = os.path.join(self.out_dir, f'{split_name}_readable_results.txt')
            with open(readable_res_out_path, 'w') as out_file:
                out_file.write(readable_results)
            logger.info(f'Saved {split_name} readable results to: {readable_res_out_path}')

        return metrics

    def eval_model_from_checkpoint(self, model_checkpoint: str, eval_splits: List[str]) -> Dict[str, Dict]:
        assert os.path.exists(model_checkpoint), f"File {model_checkpoint} does not exist"

        def init_model():
            model_args = self.get_model_init_args()
            model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, **model_args)
            return model

        trainer = self.init_trainer(self.tokenized_datasets['train'], init_model, inference_mode=True)

        return self.eval_model(trainer, eval_splits)


def init_token_classifier(model_name: str, data_splits: Dict[str, pd.DataFrame]) -> TokenClassifier:
    return TokenClassifier(model_name=model_name, data_splits=data_splits)


def get_ner_annotations_from_tagged_words(predicted_labels: List[int], predicted_labels_prob: list[float],
                                          id2label: Dict[int, str],
                                          word_tokens_to_chars, text: str) -> List:
    ner_annotations = []
    curr_entity = {}

    for i, pred in enumerate(predicted_labels):
        if not curr_entity:
            if pred == NO_ANNOTATION_LABEL:
                continue
            curr_entity = {'first_word_idx': i, 'start': word_tokens_to_chars[i]['start_char'], 'tag': id2label[pred],
                           'prob': [predicted_labels_prob[i]], 'tagged_token_id': str(uuid.uuid4())}
        if curr_entity:
            if pred == NO_ANNOTATION_LABEL:
                curr_entity['last_word_idx'] = i
                start_char = curr_entity['start']
                end_char = word_tokens_to_chars[i - 1]['end_char']
                curr_entity['end'] = end_char
                curr_entity['value'] = text[start_char:end_char]
                ner_annotations.append(curr_entity)
                curr_entity = {}
            elif id2label[pred] != curr_entity['tag']:
                curr_entity['last_word_idx'] = i
                start_char = curr_entity['start']
                end_char = word_tokens_to_chars[i - 1]['end_char']
                curr_entity['end'] = end_char
                curr_entity['value'] = text[start_char:end_char]
                ner_annotations.append(curr_entity)
                curr_entity = {'first_word_idx': i, 'start': word_tokens_to_chars[i]['start_char'],
                               'tag': id2label[pred],
                               'prob': [predicted_labels_prob[i]], 'tagged_token_id': str(uuid.uuid4())}
            else:
                curr_entity['prob'].append(predicted_labels_prob[i])

    ner_annotations_with_prob = []
    for entity in ner_annotations:
        entity['prob'] = sum(entity['prob']) / len(entity['prob']) if len(entity['prob']) > 0 else -1
        ner_annotations_with_prob.append(entity)

    return ner_annotations_with_prob


def map_word_tokens_indices_to_char_indices(text: str) -> Dict[int, Dict]:
    tokens = nltk.word_tokenize(text)
    offset = 0
    word_tokens_info = {}
    for token_idx, token in enumerate(tokens):
        offset = text.find(token, offset)
        word_tokens_info[token_idx] = {'start_char': offset, 'end_char': offset + len(token), 'text': token}
        offset += len(token)

    return word_tokens_info


def run_finetune_pipeline(data_splits: Dict[str, pd.DataFrame]):
    model_name = args.model_name
    data_splits = {'train': data_splits['train'], 'test': data_splits['eval']}
    token_classifier = init_token_classifier(model_name, data_splits)
    token_classifier.finetune_model()


def main():
    data_dir = args.data_dir
    data_splits = {'train': None, 'eval': None}
    for split_name in data_splits.keys():
        split_path = os.path.join(data_dir, f'{split_name}.csv')
        split_data = pd.read_csv(split_path, dtype={'paper_id': str})
        data_splits[split_name] = split_data

    data_splits['eval'] = data_splits['eval'].head(5)

    run_finetune_pipeline(data_splits)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--hf_token', type=str, default='huggingface_api_key')
    args = parser.parse_args()

    logger = setup_default_logger(args.output_dir)

    NO_ANNOTATION_LABEL = -1
    for label, val in id2label.items():
        if val == 'other':
            NO_ANNOTATION_LABEL = int(label)
            logger.info(f'No annotation label: {NO_ANNOTATION_LABEL}')
            break

    main()
