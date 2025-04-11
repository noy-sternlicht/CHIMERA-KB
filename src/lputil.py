import pandas as pd
from datasets import Dataset
import torch
from transformers import BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import SentenceEvaluator
from tqdm import tqdm
import statistics
import json
import string
import os
from util import word_tokenize_text
from nltk.corpus import stopwords


def build_query(context, anchor_text, relation):
    query = "Background: " + context + "\n"

    if relation == 'combination':
        query += f"Contribution: Combine '{anchor_text}' and "
    else:
        query += f"Contribution: '{anchor_text}' inspired by "

    return query


def build_query_instruct(context, anchor_text, relation):
    query = context

    if relation == 'combination':
        query += f"We suggest combining '{anchor_text}' and "
    else:
        query += f"We suggest '{anchor_text}' inspired by "

    return query


MODELS = {
    'sentence-transformers/all-mpnet-base-v2': {'query_prompt': '', 'answer_prompt': '',
                                                'query_func': build_query},
    'BAAI/bge-large-en-v1.5': {
        'query_prompt': 'Represent this sentence for searching relevant passages: ',
        'answer_prompt': '',
        'query_func': build_query_instruct},
    'intfloat/e5-large-v2': {
        'query_prompt': 'query: ',
        'answer_prompt': '',
        'query_func': build_query_instruct},
}


def read_edges(path):
    edges = pd.read_csv(path, dtype={'publication_year': int, 'source_id': str, 'target_id': str})
    return edges


def build_train_dataset(all_edges_path, train_path, model_name, nr_negatives):
    all_edges = read_edges(all_edges_path)
    train_data = read_edges(train_path)

    data = []

    build_query_func = MODELS[model_name]['query_func']
    for _, row in train_data.iterrows():
        for mode in ['tail', 'head']:
            if row['relation'] == 'inspiration' and mode == 'head':
                continue
            if row[f'{mode}_leakage'] == 'yes':
                continue

            anchor_text = row['source_text'] if mode == 'tail' else row['target_text']
            answer_text = row['target_text'] if mode == 'tail' else row['source_text']
            answer_id = row['target_id'] if mode == 'tail' else row['source_id']

            query = build_query_func(row['context'], anchor_text, row['relation'])

            data_entry = {
                'query': query,
                'answer': answer_text,
                'answer_id': answer_id,
                'label': 1
            }

            data.append(data_entry)

            anchor_id_column = 'source_id' if mode == 'tail' else 'target_id'
            answer_id_column = 'target_id' if mode == 'tail' else 'source_id'
            answer_text_column = 'target_text' if mode == 'tail' else 'source_text'
            edges_of_type = all_edges[all_edges['relation'] == row['relation']]
            positive_target_ids = edges_of_type[edges_of_type[anchor_id_column] == row[anchor_id_column]][
                answer_id_column].tolist()
            negative_examples = all_edges[~all_edges[answer_id_column].isin(positive_target_ids)]
            negative_examples = negative_examples.sample(n=nr_negatives)
            for _, neg_row in negative_examples.iterrows():
                data_entry = {
                    'query': query,
                    'answer': neg_row[answer_text_column],
                    'answer_id': neg_row[answer_id_column],
                    'label': 0
                }

                data.append(data_entry)

    dataset = pd.DataFrame(data)
    dataset = Dataset.from_pandas(dataset)

    return dataset


def domain_comparison(row):
    source_part = row['source_domain'].split('.')[0]
    target_part = row['target_domain'].split('.')[0]

    return source_part != target_part


def build_eval_dataset(eval_path: str, model_name: str):
    eval_data = read_edges(eval_path)

    data = []
    query_func = MODELS[model_name]['query_func']
    for _, row in eval_data.iterrows():
        for mode in ['tail', 'head']:
            if row['relation'] == 'inspiration' and mode == 'head':
                continue
            if row[f'{mode}_leakage'] == 'yes':
                continue

            anchor_text = row['source_text'] if mode == 'tail' else row['target_text']
            anchor_id = row['source_id'] if mode == 'tail' else row['target_id']
            answer_id = row['target_id'] if mode == 'tail' else row['source_id']
            answer_text = row['target_text'] if mode == 'tail' else row['source_text']
            query = query_func(row['context'], anchor_text, row['relation'])

            data_entry = {
                'id': f'{answer_id}_{row["id"]}',
                'context': row['context'],
                'query': query,
                'anchor_id': anchor_id,
                'anchor_text': anchor_text,
                'answer': answer_text,
                'relation': row['relation'],
                'is_cross_domain': domain_comparison(row),
                'answer_id': answer_id,
                'anchor_id_column': 'source_id' if mode == 'tail' else 'target_id',
                'answer_id_column': 'target_id' if mode == 'tail' else 'source_id',
                'label': 1
            }

            data.append(data_entry)
    dataset = pd.DataFrame(data)

    return dataset


def get_entities_data(entities_path):
    entities_data = pd.read_csv(entities_path, dtype={'entity_id': str})
    entities_data = entities_data[['entity_id', 'entity_text']]

    entity_id_to_text = dict(zip(entities_data['entity_id'], entities_data['entity_text']))

    return entity_id_to_text


def load_model(weights_precision, quantize, checkpoint, model_name):
    if weights_precision == 16:
        model_kwargs = {'torch_dtype': torch.bfloat16}
    else:
        model_kwargs = {'torch_dtype': torch.float32}
        torch.set_float32_matmul_precision('high')
    if quantize:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=getattr(torch, "bfloat16"),
            bnb_4bit_use_double_quant=True
        )

        model_kwargs = {'quantization_config': quant_config}
    if not checkpoint:
        model = SentenceTransformer(model_name, model_kwargs=model_kwargs, trust_remote_code=True)
    else:
        model = SentenceTransformer(checkpoint, model_kwargs=model_kwargs, trust_remote_code=True)
    return model


def get_eval_samples(all_edges_path, candidate_path, eval_data, entity_id_to_text):
    samples = []
    all_edges = read_edges(all_edges_path)
    all_entity_ids = []
    with open(candidate_path, 'r') as f:
        for line in f:
            all_entity_ids.append(line.strip())
    entity_id_to_text = {e: entity_id_to_text[e] for e in all_entity_ids}

    for _, example in eval_data.iterrows():
        query = example['query']
        positive_answer = example['answer']
        all_edges_of_rel_type = all_edges[all_edges['relation'] == example['relation']]
        all_positive_answers = \
            all_edges_of_rel_type[all_edges_of_rel_type[example['anchor_id_column']] == example['anchor_id']][
                example['answer_id_column']].tolist()
        candidates = [e for e in all_entity_ids if e not in all_positive_answers and e != example['anchor_id']]
        candidates.append(example['answer_id'])
        candidates = list(set(candidates))
        candidates_text = [entity_id_to_text[e] for e in candidates]

        samples.append({
            'id': example['id'],
            'query': query,
            'context': example['context'],
            'anchor_id': example['anchor_id'],
            'anchor_text': example['anchor_text'],
            'positive': positive_answer,
            'positive_id': example['answer_id'],
            'relation': example['relation'],
            'is_cross_domain': example['is_cross_domain'],
            'candidates_text': candidates_text,
            'candidates': candidates,
            'positive_index': candidates.index(example['answer_id']),
        })

    return samples, all_entity_ids


class RankingEvaluator(SentenceEvaluator):
    def __init__(
            self,
            samples,
            all_entity_ids,
            entity_id_to_text,
            model_name,
            logger,
            encode_batch_size,
            output_path,
    ):
        super().__init__()
        self.samples = samples
        self.all_entity_ids = all_entity_ids
        self.entity_id_to_text = entity_id_to_text
        self.model_name = model_name
        self.logger = logger
        self.encode_batch_size = encode_batch_size
        self.output_path = output_path

    def __call__(self, model: SentenceTransformer, output_path: str = None, epoch: int = -1, steps: int = -1) -> dict[
        str, float]:
        model_info = MODELS[self.model_name]
        self.logger.info(f'Using model: {self.model_name}')
        self.logger.info(f'Query prompt: {model_info["query_prompt"]}')
        self.logger.info(f'Answer prompt: {model_info["answer_prompt"]}')
        model = model.eval()
        with torch.no_grad():
            self.logger.info(f'Encoding {len(self.samples)} queries...')
            queries = [s['query'] for s in self.samples]
            if model_info['query_prompt']:
                self.logger.info(f'Using query prompt: {model_info["query_prompt"]}')
                queries_encoded = model.encode(queries, batch_size=self.encode_batch_size,
                                               prompt=model_info['query_prompt'])
                torch.cuda.empty_cache()
            else:
                queries_encoded = model.encode(queries, batch_size=self.encode_batch_size)
                torch.cuda.empty_cache()

            self.logger.info(f'Encoding {len(self.all_entity_ids)} entities...')
            sorted_ids = sorted(self.all_entity_ids)
            entities_sorted = [self.entity_id_to_text[i] for i in sorted_ids]
            if model_info['answer_prompt']:
                self.logger.info(f'Using answer prompt: {model_info["answer_prompt"]}')
                entities_encoded = model.encode(entities_sorted, batch_size=self.encode_batch_size,
                                                prompt=model_info['answer_prompt'])
                torch.cuda.empty_cache()
            else:
                entities_encoded = model.encode(entities_sorted, batch_size=self.encode_batch_size)
                torch.cuda.empty_cache()
            entity_id_to_encoded = {e: entities_encoded[i,] for i, e in enumerate(sorted_ids)}

        metrics = {}
        filter_types = ['combination', 'inspiration', 'cross-domain', 'all']
        metric_types = ['mr', 'mrr', 'hits@1', 'hits@3', 'hits@5', 'hits@10', 'hits@50', 'hits@100', 'median_gold_rank']
        for ftype in filter_types:
            for metric in metric_types:
                if metric != 'median_gold_rank':
                    metrics[f'{ftype}_{metric}'] = 0
                else:
                    metrics[f'{ftype}_{metric}'] = []

        def update_metrics(metrics, rank, ftype):
            metrics[f'{ftype}_mr'] += rank + 1
            metrics[f'{ftype}_mrr'] += 1 / (rank + 1)
            metrics[f'{ftype}_hits@1'] += 1 if rank < 1 else 0
            metrics[f'{ftype}_hits@3'] += 1 if rank < 3 else 0
            metrics[f'{ftype}_hits@5'] += 1 if rank < 5 else 0
            metrics[f'{ftype}_hits@10'] += 1 if rank < 10 else 0
            metrics[f'{ftype}_hits@50'] += 1 if rank < 50 else 0
            metrics[f'{ftype}_hits@100'] += 1 if rank < 100 else 0
            metrics[f'{ftype}_median_gold_rank'].append(rank + 1)

        ranker_results = []
        sanity_log = ''
        for i, sample in enumerate(tqdm(self.samples)):
            candidates_encoded = [entity_id_to_encoded[e] for e in sample['candidates']]
            similarities = model.similarity(queries_encoded[i,], candidates_encoded)
            ranked_candidates_indices = torch.argsort(similarities, descending=True).tolist()[0]
            rank = ranked_candidates_indices.index(sample['positive_index'])

            sanity_log += '====================\n'
            sanity_log += f'Query: {sample["query"]}\nPositive: {sample["positive"]}\nRank: {rank}\n'
            sanity_log += '---------------------\n'
            top_10 = ranked_candidates_indices[:10]
            for j, idx in enumerate(top_10):
                sanity_log += f'{j}. {sample["candidates_text"][idx]}\n'

            ranker_results.append({
                'id': sample['id'],
                'query': sample['query'],
                'context': sample['context'],
                'anchor_id': sample['anchor_id'],
                'anchor_text': sample['anchor_text'],
                'relation': sample['relation'],
                'positive': sample['positive'],
                'positive_id': sample['positive_id'],
                'is_cross_domain': sample['is_cross_domain'],
                'candidates': [sample['candidates'][idx] for idx in ranked_candidates_indices],
                'candidates_text': [sample['candidates_text'][idx] for idx in ranked_candidates_indices],
                'positive_index': rank})

            update_metrics(metrics, rank, 'all')
            if sample['is_cross_domain']:
                update_metrics(metrics, rank, 'cross-domain')
            if sample['relation'] == 'combination':
                update_metrics(metrics, rank, 'combination')
            if sample['relation'] == 'inspiration':
                update_metrics(metrics, rank, 'inspiration')

        for ftype in filter_types:
            nr_queries = len(metrics[f'{ftype}_median_gold_rank'])
            self.logger.info(f'Number of queries for {ftype}: {nr_queries}')
            for metric in metric_types:
                if metric == 'median_gold_rank':
                    if len(metrics[f'{ftype}_{metric}']) == 0:
                        metrics[f'{ftype}_{metric}'] = 0
                    else:
                        metrics[f'{ftype}_{metric}'] = statistics.median(metrics[f'{ftype}_{metric}'])
                elif nr_queries > 0:
                    metrics[f'{ftype}_{metric}'] /= nr_queries
                else:
                    metrics[f'{ftype}_{metric}'] = 0

        self.logger.info(json.dumps(metrics, indent=4))
        sanity_path = os.path.join(self.output_path, 'sanity.log')
        with open(sanity_path, 'w') as f:
            f.write(sanity_log)

        table_log = '\n & H@1 & H@3 & H@5 & H@10 & H@50 & H@100 & MRR & Median Rank \\\\ \n'
        for ftype in filter_types:
            table_log += f'----{ftype}\n'
            table_log += f' & {metrics[f"{ftype}_hits@1"]:.3f} & {metrics[f"{ftype}_hits@3"]:.3f} & {metrics[f"{ftype}_hits@5"]:.3f} & {metrics[f"{ftype}_hits@10"]:.3f} & {metrics[f"{ftype}_hits@50"]:.3f} & {metrics[f"{ftype}_hits@100"]:.3f} & {metrics[f"{ftype}_mrr"]:.3f} & {metrics[f"{ftype}_median_gold_rank"]:.3f} \\\\ \n'

        table_log_path = os.path.join(self.output_path, 'table.log')
        with open(table_log_path, 'w') as f:
            f.write(table_log)

        self.logger.info(f'Wrote table log to {table_log_path}')
        self.logger.info(table_log)

        results_path = os.path.join(self.output_path, 'results.json')
        results_df = pd.DataFrame(ranker_results)
        results_df.to_json(results_path, orient='records', lines=True)
        self.logger.info(f'Wrote results to {results_path}')

        return metrics


def get_model_prompts_dict(model_name):
    model_info = MODELS[model_name]
    prompts = {}
    if model_info['query_prompt']:
        prompts['query'] = model_info['query_prompt']
    if model_info['answer_prompt']:
        prompts['answer'] = model_info['answer_prompt']

    return prompts
