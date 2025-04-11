import argparse

from iter import ITERForRelationExtraction
from util import sent_tokenize_text, setup_default_logger

import os
import json
import pandas as pd
import torch
from tqdm import tqdm


def main():
    model = ITERForRelationExtraction.from_pretrained("fleonce/iter-scierc-scideberta-full").to('cuda')
    tokenizer = model.tokenizer

    test_examples = pd.read_csv(args.test_path)
    logger.info(f"Loaded {len(test_examples)} test examples")

    entities_by_rel_type = {}
    pbar = tqdm(total=len(test_examples))

    for _, example in test_examples.iterrows():
        sentences = sent_tokenize_text(example['abstract'])
        relations = []

        for sentence in sentences:
            encodings = tokenizer(sentence, return_tensors="pt").to('cuda')

            with torch.no_grad():
                generation_output = model.generate(
                    encodings["input_ids"],
                    attention_mask=encodings["attention_mask"],
                )

            sent_relations = generation_output.links[0]
            relations.extend(sent_relations)
        rel_types = model.config.link_types
        for rel in relations:
            src, rel_type, target, _ = rel
            src_txt = src[2]
            tgt_txt = target[2]
            rel_type = rel_types[int(rel_type)]
            if rel_type not in entities_by_rel_type:
                entities_by_rel_type[rel_type] = set()
            entities_by_rel_type[rel_type].add(src_txt)
            entities_by_rel_type[rel_type].add(tgt_txt)
        pbar.update(1)


    logger.info(f"Finished processing {len(test_examples)} examples")
    for rel_type, entities in entities_by_rel_type.items():
        entities_by_rel_type[rel_type] = list(entities)
        logger.info(f"Relation type: {rel_type}, Entities: {len(entities)}")

    out_path = os.path.join(args.output_path, 'entities_by_rel_type.json')
    with open(out_path, 'w') as f:
        json.dump(entities_by_rel_type, f)

    logger.info(f"Entities by relation type saved to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str)
    parser.add_argument('--output_path', type=str)

    args = parser.parse_args()

    logger = setup_default_logger(args.output_path)

    main()
