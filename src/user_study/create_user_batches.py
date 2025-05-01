import ast
import json
import uuid
import pandas as pd
import torch
from pyairtable import Api
from sentence_transformers import SentenceTransformer
from sentence_transformers import util


print('Loading data tables')

airtable_config_path = "src/user_study/airtable_config.json"
airtable_config = json.load(open(airtable_config_path))
API_KEY = open(airtable_config["key"], "r").read().strip()

api = Api(API_KEY)
annotations_table = api.table(airtable_config['annotation_table']['base_id'],
                              airtable_config['annotation_table']['table_name'])
batches_table = api.table(airtable_config['batches_table']['base_id'], airtable_config['batches_table']['table_name'])

DATA_PATH = 'human_study_examples.csv'
USERS_EXPERTISE = 'users_expertise.json'
BATCH_SIZE = 25

baselines = ['random', 'ours', 'gpt-4o', 'sciIE', 'mpnet_zero', 'positive']

print('Loading model')
model = SentenceTransformer("ibm-granite/granite-embedding-125m-english").to('cuda')


def filter_out_bad_examples(data):
    prefixes = ["The study"]
    return data[~data['context'].apply(lambda x: any(x.startswith(prefix) for prefix in prefixes))]

def divide_data_to_batches(data_path, batch_size):
    print(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)
    print(f"Loaded {len(data)} examples")

    data = data[data.apply(lambda x: len(set(x[baseline].lower() for baseline in baselines)) == len(baselines), axis=1)]
    data['arxiv_categories'] = data['arxiv_categories'].apply(ast.literal_eval)
    print(f'Loaded {len(data)} examples with all baselines')

    annotated_data = annotations_table.all()
    annotated_examples_ids = set([record['fields']['example_id'] for record in annotated_data])
    print(f"Loaded {len(annotated_examples_ids)} annotated examples")

    assigned_batches = batches_table.all()
    assigned_examples_ids = set()
    for record in assigned_batches:
        batch = json.loads(record['fields']['data'])
        batch = pd.DataFrame(batch)
        assigned_examples_ids.update(batch['id'].tolist())
    print(f"Loaded {len(assigned_examples_ids)} assigned examples")

    handled_examples = assigned_examples_ids.union(annotated_examples_ids)
    data = data[~data['id'].isin(handled_examples)]
    print(f"Loaded {len(data)} unhandled examples")

    data = filter_out_bad_examples(data)
    print(f"Filtered out bad examples, {len(data)} examples left")

    corpus_embeddings = model.encode(data['context'].tolist(), show_progress_bar=True, batch_size=1024).tolist()
    id_to_embedding = {}
    for _, row in data.iterrows():
        id_to_embedding[row['id']] = corpus_embeddings[len(id_to_embedding)]

    corpus_embeddings = id_to_embedding

    users_expertise = json.load(open(USERS_EXPERTISE))["users_expertise"]
    user_embeddings = {user: model.encode(user_info['research_areas']) for user, user_info in
                       users_expertise.items()}

    batches_by_user = {}
    used_data = set()

    for user, user_info in users_expertise.items():
        for i in range(user_info['nr_batches']):
            print(f"User {user} batch {i + 1}/{user_info['nr_batches']}")
            user_data = data[data['arxiv_categories'].apply(
                lambda x: any(category in user_info['arxiv_categories'] for category in x))]
            user_data = user_data[~user_data['id'].isin(used_data)]

            if len(user_data) == 0:
                continue

            user_data = user_data.sample(frac=1)

            embedded_user_corpus = torch.stack(
                [torch.tensor(corpus_embeddings[example_id]) for example_id in user_data['id']])
            embedded_user_query = torch.tensor(user_embeddings[user])

            batch_len = min(batch_size, len(user_data))

            batch = util.semantic_search(embedded_user_query, embedded_user_corpus, score_function=util.cos_sim,
                                         top_k=batch_len)[0]

            batch_example_ids = [batch[i]['corpus_id'] for i in range(len(batch))]
            batch = user_data.iloc[batch_example_ids]

            if user not in batches_by_user:
                batches_by_user[user] = []

            batches_by_user[user].append(batch)
            used_data.update(batch['id'].tolist())

    return batches_by_user


def main():
    print('Creating user batches')
    data_batches = divide_data_to_batches(DATA_PATH, BATCH_SIZE)
    batches_records = batches_table.all()

    for user, batches in data_batches.items():
        user_batches = [record for record in batches_records if record['fields']['annotator'] == user]
        highest_priority = [int(record['fields']['priority']) for record in user_batches]
        highest_priority = max(highest_priority) if highest_priority else -1
        for i, batch in enumerate(batches):
            batch_id = str(uuid.uuid4())

            batches_table.create({
                "priority": str(highest_priority + i + 1),
                "batch_id": batch_id,
                "data": json.dumps(batch.to_dict(orient='records')),
                "annotator": user,
                "status": "not_started"
            })


if __name__ == '__main__':
    main()
