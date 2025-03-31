import argparse
import json

import pandas as pd
from util import setup_default_logger, preprocess_split_selected_entity_types, RelationEntity, compute_entity_agreement


def fix_entity_types(readable_entities):
    if 'analogy-src' in readable_entities:
        readable_entities['inspiration-src'] = readable_entities['analogy-src']
        del readable_entities['analogy-src']
    if 'analogy-target' in readable_entities:
        readable_entities['inspiration-target'] = readable_entities['analogy-target']
        del readable_entities['analogy-target']
    return readable_entities


def main():
    eval_examples = pd.read_csv(args.eval_data_path, dtype={'paper_id': str})
    eval_examples = preprocess_split_selected_entity_types(eval_examples,
                                                           ['comb-element', 'analogy-src', 'analogy-target'])
    pure_predictions = pd.read_json(args.pure_results_path, dtype={'paper_id': str})

    pred_entities = []
    gold_entities = []
    texts = []

    for _, example in eval_examples.iterrows():
        texts.append(example['text'])
        gold = json.loads(example['readable_entities'])
        gold = fix_entity_types(gold)

        doc_gold_entities = []
        for tag, entities in gold.items():
            for entity in entities:
                doc_gold_entities.append(RelationEntity('', tag, entity, []))

        gold_entities.append(doc_gold_entities)
        pred = pure_predictions[pure_predictions['paper_id'] == example['paper_id']]['readable_entities'].values[0]
        if not pred or pred == '{}':
            pred = {}
        else:
            pred = json.loads(pred)

        pred = fix_entity_types(pred)
        doc_pred_entities = []
        for tag, entities in pred.items():
            for entity in entities:
                doc_pred_entities.append(RelationEntity('', tag, entity, []))

        pred_entities.append(doc_pred_entities)

    entity_agreement = compute_entity_agreement(gold_entities, pred_entities, texts,
                                                ['comb-element', 'inspiration-src', 'inspiration-target'], logger)

    logger.info('Entity agreement: %s', entity_agreement)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pure_results_path', help='Path to the JSON configuration file')
    parser.add_argument('--eval_data_path', type=str, required=True, help='statements to be evaluated')
    parser.add_argument('--output_path', type=str, required=True, help='output path')

    args = parser.parse_args()

    logger = setup_default_logger(args.output_path)

    main()
