import argparse
import json
import os

import pandas as pd
from tqdm import tqdm
from util import setup_default_logger

from eval import Relation, RelationEntity, compute_entity_agreement, eval_rel_extraction


def main():
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    eval_data = pd.read_csv(args.eval_path, dtype={'paper_id': str})
    predictions = json.load(open(args.predictions_path))
    # print(predictions)

    pred_rels = []
    pred_entities = []
    gold_rels = []
    gold_entities = []
    texts = []

    pbar = tqdm(total=len(eval_data), desc="Evaluating...")

    for _, example in eval_data.iterrows():
        texts.append(example['text'])
        paper_id = example['paper_id']

        pred_relation = predictions.get(paper_id, {})
        extracted_relations_pred = {}
        example_pred_entities = []
        if pred_relation:
            pred_rel_type = list(pred_relation.keys())[0]
            pred_rel_entities = pred_relation[pred_rel_type][0]
            relation = Relation.from_entity_dictionaries(pred_rel_type, pred_rel_entities, {})
            extracted_relations_pred[relation.relation_type] = [relation]
            for entity_type, entities in pred_rel_entities.items():
                example_pred_entities.extend([RelationEntity('', entity_type, ent, []) for ent in entities])

        pred_rels.append(extracted_relations_pred)
        pred_entities.append(example_pred_entities)

        gold_recomb = json.loads(example['readable_relations'])
        if gold_recomb:
            rel_type = list(gold_recomb.keys())[0]
            rel_entities = gold_recomb[rel_type][0]
            if rel_type == 'analogy':
                rel_type = 'inspiration'

            if 'analogy-src' in rel_entities:
                rel_entities['inspiration-src'] = rel_entities.pop('analogy-src')
            if 'analogy-target' in rel_entities:
                rel_entities['inspiration-target'] = rel_entities.pop('analogy-target')
            gold_recomb = {'type': rel_type, 'entities': rel_entities}
        else:
            gold_recomb = {}

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
                                                        ['comb-element', 'inspiration-src',
                                                         'inspiration-target'],
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

    logger.info(f"\n-----\nGoLLIE Results: {json.dumps(total_results, indent=4)}\n-----")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--eval_path', type=str)
    parser.add_argument('--predictions_path', type=str)

    args = parser.parse_args()
    logger = setup_default_logger(args.output_dir)
    main()
