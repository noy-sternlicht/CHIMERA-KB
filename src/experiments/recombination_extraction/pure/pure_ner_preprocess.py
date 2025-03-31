import argparse
import json
import os.path
from typing import List, Dict, Tuple
from util import word_tokenize_text, map_chars_into_words, sent_tokenize_text, preprocess_split_selected_entity_types
import pandas as pd
import jsonlines
from tqdm import tqdm

NerAnnotation = list[str]


def find_entity_word_indices(entity: Dict, char_to_word_index: Dict[int, int], text_words: List[str]) \
        -> Tuple[int, int]:
    entity_first_word_index = -1
    entity_first_word_char_index = -1
    for i in range(entity['start'], entity['end']):
        if i in char_to_word_index:
            entity_first_word_index = char_to_word_index[i]
            entity_first_word_char_index = i
            break
    if entity_first_word_index == -1:
        raise ValueError(f'Tokenization error: Unable to find [entity={entity["value"]}] in the text.')

    entity_last_word_index = entity_first_word_index
    for i in range(entity['end'] - 1, entity_first_word_char_index - 1, -1):
        if i in char_to_word_index:
            entity_last_word_index = char_to_word_index[i]
            break

    assertion_msg_info = f'[entity={entity["value"]}],' \
                         f'[first_word_index={entity_first_word_index}], ' \
                         f'[last_word_index={entity_last_word_index}]'

    assert entity_first_word_index <= entity_last_word_index, f'Entity word indices are invalid: {assertion_msg_info}'

    assert 0 <= entity_first_word_index and entity_last_word_index < len(
        text_words), f'Entity word indices are out of bounds: {assertion_msg_info}'

    return entity_first_word_index, entity_last_word_index


def format_ner(entities: List[Dict], char_to_word_index: Dict[int, int], text_words: List[str]) -> List[NerAnnotation]:
    formatted_ner = []
    for entity in entities:
        entity_first_word_index, entity_last_word_index = find_entity_word_indices(entity,
                                                                                   char_to_word_index,
                                                                                   text_words)
        formatted_ner.append([entity_first_word_index, entity_last_word_index, entity['tag'], entity['value']])
    return formatted_ner


def get_sentence_ner_annotations(formatted_ner: List[NerAnnotation], sentence_start: int, sentence_end: int) \
        -> List[NerAnnotation]:
    sentence_ner = []
    assert sentence_start <= sentence_end, f'Invalid sentence: [{sentence_start}, {sentence_end}]'
    for ner in formatted_ner:
        ner_start = int(ner[0])
        ner_end = int(ner[1])
        assert ner_start <= ner_end, f'Invalid ner: {ner}'
        if ner_start >= sentence_start and ner_end < sentence_end:  # ner is inside the sentence
            sentence_ner.append(ner)
        elif ner_start < sentence_start and ner_end >= sentence_end:  # sentence is inside the ner
            print(f'Warning: Sentence is inside the ner: {ner}')
            sentence_ner.append([sentence_start, sentence_end, ner[2]])
        elif sentence_start <= ner_start < sentence_end <= ner_end:  # ner starts inside the sentence and ends after
            print(f'Warning: Ner starts inside the sentence and ends after: {ner}')
            sentence_ner.append([ner_start, sentence_end, ner[2]])
        elif ner_start < sentence_start <= ner_end < sentence_end:  # ner starts before the sentence and ends inside
            print(f'Warning: Ner starts before the sentence and ends inside: {ner}')
            sentence_ner.append([sentence_start, ner_end, ner[2]])
    for ner in sentence_ner:
        assert sentence_start <= int(ner[0]) and int(ner[1]) <= sentence_end, f'Invalid ner: {ner}'
    return sentence_ner


def get_sentence_formatted_relations(relations, formatted_ner):
    sentence_entities_by_value = {}
    for ner in formatted_ner:
        sentence_entities_by_value[ner[-1]] = ner

    sentence_relations = []
    for rel_type, relations in relations.items():
        if rel_type == 'analogy':
            relation = relations[0]
            src_entity = relation['analogy-src'][0]
            target_entity = relation['analogy-target'][0]
            if src_entity in sentence_entities_by_value and target_entity in sentence_entities_by_value:
                src_ner = sentence_entities_by_value[src_entity][:2]
                target_ner = sentence_entities_by_value[target_entity][:2]
                sentence_relations.append(src_ner + target_ner + [rel_type.upper()])
        if rel_type == 'combination':
            relation = relations[0]
            combination_elements = relation['comb-element']
            seen = set()
            for val_1, entity_1 in sentence_entities_by_value.items():
                for val_2, entity_2 in sentence_entities_by_value.items():
                    if val_1 != val_2 and val_1 in combination_elements and val_2 in combination_elements:
                        if (val_1, val_2) in seen or (val_2, val_1) in seen:
                            continue
                        sentence_relations.append(entity_1[:2] + entity_2[:2] + [rel_type.upper()])
                        seen.add((val_1, val_2))
    return sentence_relations


def preprocess_split(data: pd.DataFrame, output_path: str):
    results = []
    pbar = tqdm(total=len(data))
    bad_entries = 0
    for i, row in data.iterrows():
        res = {
            'doc_key': row['paper_id'],
            'sentences': [],
            'ner': [],
            'relations': [],
        }

        text = row['text']
        text_words = word_tokenize_text(text)
        char_to_word_index = map_chars_into_words(text_words, text)
        entity_annotations = json.loads(row['entities'])
        relation_annotations = json.loads(row['readable_relations'])
        all_formatted_ner = format_ner(entity_annotations, char_to_word_index, text_words)
        formatted_ner_by_sentence = []
        formatted_relations_by_sentence = []

        sentences = sent_tokenize_text(text)
        sentences_words = []
        curr_sentence_start = 0
        nr_rels = 0
        for sentence in sentences:
            tokenized_sentence = word_tokenize_text(sentence)
            if len(tokenized_sentence) == 0:
                continue
            sentences_words.append(tokenized_sentence)
            curr_sentence_end = curr_sentence_start + len(tokenized_sentence)
            sentence_ner = get_sentence_ner_annotations(all_formatted_ner, curr_sentence_start, curr_sentence_end)
            formatted_ner_by_sentence.append(sentence_ner)
            sentence_relations = get_sentence_formatted_relations(relation_annotations, sentence_ner)
            formatted_relations_by_sentence.append(sentence_relations)
            if len(sentence_relations) > 0:
                nr_rels += len(sentence_relations)
            curr_sentence_start = curr_sentence_end

        assert len(sentences_words) == len(formatted_ner_by_sentence)
        res['sentences'] = sentences_words
        res['ner'] = formatted_ner_by_sentence
        res['relations'] = formatted_relations_by_sentence

        if relation_annotations and not nr_rels:
            print(f'Warning: No relations found in the text: {row["paper_id"]}')
            bad_entries += 1

        results.append(res)
        pbar.update(1)

    if bad_entries:
        print(f'Warning: {bad_entries} / {len(data)} entries have no relations')

    with jsonlines.open(output_path, 'w') as f:
        f.write_all(results)


def main(data_dir: str, output_dir: str, entity_types: List[str]):
    assert os.path.exists(data_dir), f'Path {data_dir} does not exist.'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    out_names_mapping = {
        'train': 'train.json',
        'valid': 'dev.json',
        'eval': 'test.json',
    }

    for split in ['train', 'valid', 'eval']:
        print(f'Processing {split} split')
        split_data = pd.read_csv(os.path.join(data_dir, f'{split}.csv'), dtype={'paper_id': str})
        split_data = preprocess_split_selected_entity_types(split_data, entity_types)
        preprocess_split(split_data, os.path.join(output_dir, out_names_mapping[split]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        help='Path to the data directory containing the data splits: train.csv. valid.csv, test.csv')
    parser.add_argument('--output_dir', help='Path to the output directory where the preprocessed data will be saved.')
    parser.add_argument('--entity_types', type=str, nargs='+', required=False,
                        help='Entity types to evaluate', default=['comb-element', 'analogy-src', 'analogy-target'])
    args = parser.parse_args()
    main(args.data_dir, args.output_dir, args.entity_types)
