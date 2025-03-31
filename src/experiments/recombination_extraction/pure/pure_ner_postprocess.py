import argparse
import json
import os
import uuid
from typing import Dict
from util import spans_overlap, preprocess_split_selected_entity_types
import pandas as pd
import nltk


def map_word_tokens_indices_to_char_indices(text: str) -> Dict[int, Dict]:
    tokens = nltk.word_tokenize(text)
    offset = 0
    word_tokens_info = {}
    for token_idx, token in enumerate(tokens):
        offset = text.find(token, offset)
        word_tokens_info[token_idx] = {'start_char': offset, 'end_char': offset + len(token), 'text': token}
        offset += len(token)

    return word_tokens_info


def main(prediction_path: str, papers_path: str, output_dir: str, entity_types: list):
    pure_annotations = pd.read_json(prediction_path, lines=True, dtype={'doc_key': str})[['doc_key', 'predicted_ner']]
    formatted_results = []
    papers_data = pd.read_csv(papers_path, dtype={'paper_id': str})
    papers_data = preprocess_split_selected_entity_types(papers_data, entity_types)

    for _, row in pure_annotations.iterrows():
        paper_id = row['doc_key']
        result = {'paper_id': paper_id,
                  'text': '',
                  'document_class': 'irrelevant',
                  'entities': [],
                  'relations': [],
                  'readable_entities': {},
                  }

        paper_info = papers_data[papers_data['paper_id'] == paper_id]
        if paper_info.empty:
            print(f'Warning: Paper {paper_id} not found in the papers data')
            continue
        text = paper_info.iloc[0]['text']
        result['text'] = text
        word_tokens_to_chars = map_word_tokens_indices_to_char_indices(text)
        all_predicted_entities = []
        for sent_predictions in row['predicted_ner']:
            all_predicted_entities.extend(sent_predictions)

        verified_entities = []
        for entity in all_predicted_entities:
            start_word_idx = entity[0]
            end_word_idx = entity[1]
            entity_prob = entity[3]
            start_char = word_tokens_to_chars[start_word_idx]['start_char']
            end_char = word_tokens_to_chars[end_word_idx]['end_char']
            tag = entity[2]
            new_entity = {'tagged_token_id': str(uuid.uuid4()), 'tag': tag,
                          'start': start_char, 'end': end_char, 'prob': entity_prob, 'first_word_idx': start_word_idx,
                          'last_word_idx': end_word_idx, 'text': text[start_char:end_char]}
            overlap = False
            for existing_entity in verified_entities:
                if spans_overlap(new_entity, existing_entity):
                    overlap = True
                    break
            if not overlap:
                verified_entities.append(new_entity)
        result['entities'] = json.dumps(verified_entities)
        if len(verified_entities) > 0:
            result['document_class'] = 'relevant'
            readable_entities = {}
            for entity in verified_entities:
                if entity['tag'] not in readable_entities:
                    readable_entities[entity['tag']] = []
                readable_entities[entity['tag']].append(entity['text'])
            result['readable_entities'] = json.dumps(readable_entities)
        formatted_results.append(result)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    out_path = os.path.join(output_dir, 'formatted_pure_predictions.json')
    pd.DataFrame(formatted_results).to_json(out_path, orient='records')
    print(f"Saved the formatted results to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--papers_path', help='Path to the JSON file containing the papers')
    parser.add_argument('--prediction_path', help='Path to the JSON file containing the predictions')
    parser.add_argument('--entity_types', nargs='+', type=str, required=False,
                        help='Entity types to evaluate', default=['comb-element', 'analogy-src', 'analogy-target'])
    parser.add_argument('--output_dir', help='Path to the output directory where the processed data will be saved.')
    args = parser.parse_args()
    main(args.prediction_path, args.papers_path, args.output_dir, args.entity_types)
