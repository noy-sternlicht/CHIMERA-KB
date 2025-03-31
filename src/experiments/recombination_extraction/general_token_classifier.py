import argparse
import json
import os
import uuid
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np
import torch
from huggingface_hub import login
from peft import LoraConfig, TaskType, PeftModel
from scipy.special import softmax

import pandas as pd
import nltk
import transformers
from datasets import Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, EvalPrediction, BitsAndBytesConfig
from util import sent_tokenize_text, word_tokenize_text, map_chars_into_words, \
    setup_default_logger, calculate_entity_bertscore_at_percent, preprocess_split_for_abstract_only, preprocess_split_selected_entity_types, RelationEntity, compute_entity_agreement


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
    def __init__(self, model_name: str, model_conf: Dict, id2label: Dict[int, str],
                 data_splits: Dict[str, pd.DataFrame]):
        login(token="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")
        self.id2label = id2label
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.model_name = model_name
        self.tokenizer = self.init_tokenizer()
        self.data_collator = transformers.DataCollatorForTokenClassification(tokenizer=self.tokenizer, padding=True)
        self.tokenized_datasets = self.init_datasets(data_dfs=data_splits)

        self.run_name = f"{self.model_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        self.out_dir = os.path.join(model_conf['out_dir'], self.run_name)
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

    def init_tokenizer(self):
        tokenizer_init_args = {"trust_remote_code": True, "add_prefix_space": True}
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, **tokenizer_init_args)
        if not hasattr(tokenizer, 'model_max_length') or tokenizer.model_max_length is None:
            tokenizer.model_max_length = 512
        # tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    def get_model_init_args(self, qlora=False) -> Dict:

        model_init_args = {"device_map": None,
                           "pad_token_id": self.tokenizer.pad_token_id,
                           "num_labels": len(self.id2label),
                           "id2label": self.id2label,
                           "label2id": self.label2id,
                           }

        if qlora:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=getattr(torch, "float16"),
                bnb_4bit_use_double_quant=True
            )

            model_init_args["quant_config"] = quant_config
            model_init_args["torch_dtype"] = torch.bfloat16,

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

    def init_trainer(self, train_dataset: Dataset, model_init, inference_mode: bool,
                     qlora=False) -> transformers.Trainer:
        peft_params = LoraConfig(
            r=64,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            inference_mode=inference_mode,
            task_type=TaskType.TOKEN_CLS,
        )

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

        if qlora:
            def qlora_model_init():
                model = model_init()
                model = PeftModel(model, peft_params)
                model.print_trainable_parameters()
                return model

            model_init_func = model_init if inference_mode else qlora_model_init
            training_args.bf16 = True

        else:
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

    def init_hpo_trainer(self, training_dataset: Dataset, eval_dataset: Dataset) -> transformers.Trainer:
        training_args = transformers.TrainingArguments(output_dir=self.out_dir,
                                                       gradient_checkpointing=True,
                                                       gradient_accumulation_steps=4,
                                                       gradient_checkpointing_kwargs={"use_reentrant": False},
                                                       optim="paged_adamw_8bit",
                                                       per_device_train_batch_size=1,
                                                       num_train_epochs=10,
                                                       weight_decay=0.1,
                                                       lr_scheduler_type="cosine",
                                                       learning_rate=6.e-5,
                                                       logging_steps=10,
                                                       run_name="test",
                                                       warmup_ratio=0.1
                                                       )

        def model_init(trial):
            model_args = self.get_model_init_args()
            return AutoModelForTokenClassification.from_pretrained(self.model_name, **model_args)

        trainer = transformers.Trainer(
            model=None,
            model_init=model_init,
            args=training_args,
            data_collator=self.data_collator,
            train_dataset=training_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )
        return trainer

    def train_with_hpo(self, nr_trials: int) -> Tuple[str, Dict]:
        logger.info(f'Starting hyperparameter optimization with {nr_trials} trials')

        def optuna_hp_space(trial):
            return {
                "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
                "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size",
                                                                         [1, 2, 4, 8]),
                "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 10),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            }

        def compute_objective(metrics: Dict[str, float]) -> float:
            labels_precision = []
            labels_recall = []

            for label_name in self.id2label.values():
                if label_name == 'other':
                    continue
                label_precision = metrics[f'eval_{label_name}_precision']
                if label_precision != -1:
                    labels_precision.append(label_precision)
                label_recall = metrics[f'eval_{label_name}_recall']
                if label_recall != -1:
                    labels_recall.append(label_recall)

            avg_precision = sum(labels_precision) / len(labels_precision) if len(labels_precision) > 0 else -1
            avg_recall = sum(labels_recall) / len(labels_recall) if len(labels_recall) > 0 else -1

            if avg_precision == -1 or avg_recall == -1:
                return 0

            avg_f1 = 2 * avg_precision * avg_recall / (
                    avg_precision + avg_recall) if avg_precision + avg_recall > 0 else 0
            return avg_f1

        trainer = self.init_hpo_trainer(self.tokenized_datasets['train'], self.tokenized_datasets['valid'])

        hpo_res = trainer.hyperparameter_search(
            direction="maximize",
            backend="optuna",
            hp_space=optuna_hp_space,
            compute_objective=compute_objective,
            n_trials=nr_trials,
        )

        best_run_params = hpo_res.hyperparameters
        logger.info(f'Best run params: {best_run_params}')

        trainer.args.learning_rate = best_run_params['learning_rate']
        trainer.args.weight_decay = best_run_params['weight_decay']
        trainer.args.num_train_epochs = best_run_params['num_train_epochs']
        trainer.args.per_device_train_batch_size = best_run_params['per_device_train_batch_size']

        trainer.train()
        best_checkpoint_path = os.path.join(self.out_dir, f'best_checkpoint_{self.run_name}')
        trainer.save_model(best_checkpoint_path)
        logger.info(f'Model saved to {best_checkpoint_path}')

        # metrics = self.eval_model(trainer, ['test'])

        return best_checkpoint_path, {}

    def finetune_model(self, at_k_values=(0.25, 0.5, 1.0)) -> Tuple[str, Dict[str, Dict]]:
        def init_model():
            model_args = self.get_model_init_args()
            return AutoModelForTokenClassification.from_pretrained(self.model_name, **model_args)

        trainer = self.init_trainer(self.tokenized_datasets['train'], init_model, inference_mode=False)
        trainer.train()
        best_checkpoint_path = os.path.join(self.out_dir, f'best_checkpoint_{self.run_name}')
        trainer.save_model(best_checkpoint_path)
        logger.info(f'Saved best checkpoint to: {best_checkpoint_path}')

        metrics = self.eval_model(trainer, ['test'], at_k_values)
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

        # parsed_results = postprocess_entity_predictions(parsed_results, self.out_dir)
        return parsed_results

    def eval_model(self, trainer, eval_splits, at_k_values: List[float]) -> Dict:
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

            
            entity_results = compute_entity_agreement(pred_ent_objects, docs_gold_ents_objects, texts, list(self.id2label.values()), logger)
            print(f'Entity results: {entity_results}')

            for k in at_k_values:
                bertscore_at_k = calculate_entity_bertscore_at_percent(docs_pred_ents, docs_gold_ents, texts,
                                                                       list(self.id2label.values()), percent=k)
                readable_results += f'BERT score at {k}: {bertscore_at_k}\n'
                for ent_type, ent_scores in bertscore_at_k.items():
                    for metric_name, metric_value in ent_scores.items():
                        metrics[split_name].update({f'{ent_type}_{metric_name}_at_{k}': metric_value})
            readable_res_out_path = os.path.join(self.out_dir, f'{split_name}_readable_results.txt')
            with open(readable_res_out_path, 'w') as out_file:
                out_file.write(readable_results)
            logger.info(f'Saved {split_name} readable results to: {readable_res_out_path}')
            logger.info(f'Metrics: {metrics}')

        return metrics

    def eval_model_from_checkpoint(self, model_checkpoint: str, eval_splits: List[str],
                                   at_k_values=(0.25, 0.5, 1.0)) -> Dict[str, Dict]:
        assert os.path.exists(model_checkpoint), f"File {model_checkpoint} does not exist"

        def init_model():
            model_args = self.get_model_init_args()
            model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, **model_args)
            return model

        trainer = self.init_trainer(self.tokenized_datasets['train'], init_model, inference_mode=True)

        return self.eval_model(trainer, eval_splits, at_k_values)


def init_token_classifier(model_name: str, model_conf_path: str, id2label: Dict[int, str],
                          data_splits: Dict[str, pd.DataFrame], epoch_nr=1) -> TokenClassifier:
    assert os.path.exists(model_conf_path), f"File {model_conf_path} does not exist"
    model_conf = json.load(open(model_conf_path, 'r'))[model_name]
    model_conf['out_dir'] = os.path.join(model_conf['out_dir'], f'epoch_{epoch_nr}')
    return TokenClassifier(model_name=model_name, model_conf=model_conf, id2label=id2label, data_splits=data_splits)


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


def run_finetune_pipeline(config: Dict, data_splits: Dict[str, pd.DataFrame]):
    run_hpo = config['run_hpo']
    nr_trials = config['nr_hpo_trials']
    model_name = config['model_name']
    model_conf_path = config['models_configs_path']
    id2label = config['id2label']
    id2label = {int(k): v for k, v in id2label.items()}
    if config['synth_data']['enabled']:
        all_data = pd.concat([data_splits['train'], data_splits['valid'], data_splits['test']])
        synth_data = get_synth_data(config)
        all_data = all_data[~all_data['paper_id'].isin(synth_data['paper_id'])].sample(frac=1).reset_index(drop=True)
        data_splits['valid'] = all_data.iloc[:len(all_data) // 2].sample(frac=1).reset_index(drop=True)
        data_splits['test'] = all_data.iloc[len(all_data) // 2:].sample(frac=1).reset_index(drop=True)
        data_splits['train'] = synth_data.sample(frac=1).reset_index(drop=True)

    if run_hpo:
        token_classifier = init_token_classifier(model_name, model_conf_path, id2label, data_splits)
        token_classifier.train_with_hpo(nr_trials)
    else:
        data_splits = {'train': data_splits['train'], 'test': data_splits['eval']}
        token_classifier = init_token_classifier(model_name, model_conf_path, id2label, data_splits)
        checkpoint, metrics = token_classifier.finetune_model(config['eval_at_k'])
        logger.info(f'Finetuning metrics: {metrics}')


def get_synth_data(config: Dict) -> pd.DataFrame:
    synth_data = pd.read_csv(config['synth_data']['synth_data_path'], dtype={'paper_id': str})
    synth_data_seed = synth_data[~synth_data['paper_id'].str.contains('synth')]
    non_seed_synth_data = synth_data[synth_data['paper_id'].str.contains('synth')]
    sample_size = config['synth_data']['nr_synth_examples']
    logger.info(f'Loaded {len(non_seed_synth_data)} non-seed synthetic examples')
    non_seed_synth_data = non_seed_synth_data.sample(sample_size)
    logger.info(f'Sampled {len(non_seed_synth_data)} non-seed synthetic examples')

    synth_data = pd.concat([synth_data_seed, non_seed_synth_data])
    logger.info(f'Loaded {len(synth_data)} synthetic examples')
    return synth_data


def run_cross_val_pipeline(config: Dict, data_splits: Dict[str, pd.DataFrame]):
    cross_val_fold_num = config['k_fold_cross_val']
    model_name = config['model_name']
    model_conf_path = config['models_configs_path']
    id2label = config['id2label']
    id2label = {int(k): v for k, v in id2label.items()}
    all_data = pd.concat([data_splits['train'], data_splits['valid'], data_splits['test']])
    all_data = all_data.sample(frac=1).reset_index(drop=True)
    synth_data = pd.DataFrame()
    pre_synth_fold_size = len(all_data) // cross_val_fold_num
    if config['synth_data']['enabled']:
        synth_data = get_synth_data(config)
        all_data = all_data[~all_data['paper_id'].isin(synth_data['paper_id'])]
        all_data = all_data.sample(frac=1).reset_index(drop=True)

    after_synth_fold_size = len(all_data) // cross_val_fold_num
    nr_missing_test_examples = pre_synth_fold_size - after_synth_fold_size
    metrics = []
    out_text = ""

    for fold_num in range(cross_val_fold_num):
        logger.info(f'Running fold {fold_num + 1}/{cross_val_fold_num}')
        fold_data = all_data.iloc[fold_num * after_synth_fold_size:(fold_num + 1) * after_synth_fold_size]
        other_data = all_data.drop(fold_data.index)
        if nr_missing_test_examples > 0:
            other_data, missing_test_examples = other_data.iloc[:-nr_missing_test_examples], other_data.iloc[
                                                                                             -nr_missing_test_examples:]
            fold_data = pd.concat([fold_data, missing_test_examples])
        other_data = pd.concat([other_data, synth_data]).sample(frac=1).reset_index(drop=True)
        data_splits = {'train': other_data, 'test': fold_data}
        token_classifier = init_token_classifier(model_name, model_conf_path, id2label, data_splits, epoch_nr=fold_num)
        _, fold_metrics = token_classifier.finetune_model()
        metrics.append(fold_metrics['test'])

    aggregated_metrics = {}
    for fold_metrics in metrics:
        for metric_name, metric_value in fold_metrics.items():
            aggregated_metrics[metric_name] = aggregated_metrics.get(metric_name, [])
            if metric_value != -1:
                aggregated_metrics[metric_name].append(metric_value)

    for metric_name, metric_values in aggregated_metrics.items():
        aggregated_metrics[metric_name] = np.mean(metric_values)

    logger.info(f'Aggregated metrics: {aggregated_metrics}')
    out_text += json.dumps(aggregated_metrics) + '\n'

    results_path = os.path.join(config['output_dir'], 'cross_val_results.txt')
    logger.info(f'Saving cross validation results to: {results_path}')
    with open(results_path, 'w') as out_file:
        out_file.write(out_text)


def main(config: Dict):
    data_dir = config['data_dir']
    data_splits = {'train': None, 'valid': None, 'test': None, 'eval': None}
    for split_name in data_splits.keys():
        split_path = os.path.join(data_dir, f'{split_name}.csv')
        split_data = pd.read_csv(split_path, dtype={'paper_id': str})
        selected_entity_types = config.get('id2label', {}).values()
        selected_entity_types = [entity_type for entity_type in selected_entity_types if entity_type != 'other']
        data_splits[split_name] = preprocess_split_selected_entity_types(split_data, selected_entity_types)

    cross_val_fold_num = config.get('k_fold_cross_val', 1)
    if cross_val_fold_num == 1:
        run_finetune_pipeline(config, data_splits)
    else:
        run_cross_val_pipeline(config, data_splits)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', help='Path to the JSON configuration file')
    args = parser.parse_args()

    assert os.path.exists(args.config_path)
    config_dict = json.load(open(args.config_path, 'r'))
    out_path = config_dict['output_dir']
    logger = setup_default_logger(out_path)

    NO_ANNOTATION_LABEL = -1
    for label, val in config_dict['id2label'].items():
        if val == 'other':
            NO_ANNOTATION_LABEL = int(label)
            logger.info(f'No annotation label: {NO_ANNOTATION_LABEL}')
            break


    main(config_dict)
