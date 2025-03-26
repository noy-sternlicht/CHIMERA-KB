import json
import time
import uuid
from typing import List, Dict
from openai import OpenAI

from tqdm import tqdm

SPAN_COMPARISON_PROMPT = """You are tasked with comparing two spans extracted from a scientific text to determine if they discuss the same {ENTITY_TYPE}. Follow these instructions carefully:

1. First, read the full text for context:
<full_text>
{TEXT}
</full_text>

2. Now, consider these two spans extracted from the text above:
<span1>
{SPAN1}
</span1>

<span2>
{SPAN2}
</span2>

3. Your task is to carefully analyze these two spans and determine if they discuss the same {ENTITY_TYPE}. The idea the spans discuss should be exactly the same, up to minor lexical or semantic variations.

4. In your analysis, consider the following:
   a. The main topic or idea presented in each span
   b. The context in which these spans appear in the full text
   c. Any potential contradictions between the spans

5. After your analysis, provide a justification for your determination. Explain your reasoning clearly, referencing specific elements from the spans and the full text if necessary.

6. Based on your analysis and justification, provide a "Yes" or "No" answer to whether the spans discuss the same {ENTITY_TYPE}.

7. Present your response in the following format:
   <justification>
   [Your detailed justification here]
   </justification>

   <answer>
   [Your "Yes" or "No" answer here]
   </answer>"""

OPEN_AI_KEY = open('openai_key').read().strip()
OPEN_AI_CLIENT = OpenAI(api_key=OPEN_AI_KEY)


def prompt_span_comparison_model(span1: str, span2: str, text: str, span_type: str, client="gpt") -> str:
    prompt = SPAN_COMPARISON_PROMPT.format(TEXT=text, SPAN1=span1, SPAN2=span2, ENTITY_TYPE=span_type)

    start_time = time.time()
    completion = OPEN_AI_CLIENT.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=400,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    response = completion.choices[0].message.content
    print(f"Time taken: {time.time() - start_time}")

    res_start = response.find("<answer>") + len("<answer>")
    res_end = response.find("</answer>")
    if res_start == -1 or res_end == -1:
        return ''

    answer = response[res_start:res_end].strip()
    if answer.lower() not in ['yes', 'no']:
        return ''

    return answer.lower()


class RelationEntity:
    def __init__(self, tagged_token_id: str, entity_type: str, value: str, coreferences: List[str]):
        if not tagged_token_id:
            tagged_token_id = str(uuid.uuid4())

        self.id = tagged_token_id
        self.type = entity_type

        if value not in coreferences:
            coreferences.append(value)

        self.value = value
        self.coreferences = coreferences

    def compute_similarity_LLM(self, other: 'RelationEntity', paper_text: str):
        max_score = 0
        max_match = None

        if self.type != other.type:
            return max_match, max_score

        for coref in self.coreferences:
            for other_coref in other.coreferences:
                match1 = prompt_span_comparison_model(coref, other_coref, paper_text, self.type)
                match2 = prompt_span_comparison_model(other_coref, coref, paper_text, self.type)
                if match1 == 'yes' and match2 == 'yes':
                    max_match = other_coref
                    max_score = 1
                    break

        return max_match, max_score

    def get_most_similar_entity_LLM(self, entities: List['RelationEntity'], paper_text: str):
        max_score = 0
        most_similar_span = None
        max_match = None
        for entity in entities:
            span, score = self.compute_similarity_LLM(entity, paper_text)
            if score > max_score:
                max_score = score
                max_match = entity
                most_similar_span = span
        return max_match, most_similar_span, max_score

    def __str__(self):
        return self.value


class Relation:
    def __init__(self, relation_type: str, entities: List['RelationEntity']):
        self.id = str(uuid.uuid4())
        self.relation_type = relation_type
        self.entities = entities
        self.nr_entities = len(entities)
        self.entities_by_type = {}
        for entity in entities:
            if entity.type not in self.entities_by_type:
                self.entities_by_type[entity.type] = []
            self.entities_by_type[entity.type].append(entity)

    @classmethod
    def from_entity_dictionaries(cls, relation_type: str, rel_entities_by_type: Dict[str, str],
                                 entity_coreferences: Dict[str, List[str]]):
        rel_entities = []
        for entity_type, entity_values in rel_entities_by_type.items():
            for entity_val in entity_values:
                rel_entities.append(RelationEntity(str(uuid.uuid4()), entity_type, entity_val,
                                                   entity_coreferences.get(entity_val, [])))
        return cls(relation_type, rel_entities)

    def compute_similarity(self, other: 'Relation', paper_text: str, similarity_threshold: float):
        similarity_score = 0
        entity_matches = {}
        for entity_type, entities in self.entities_by_type.items():
            if entity_type not in entity_matches:
                entity_matches[entity_type] = {}
            for entity in entities:
                entity_matches[entity_type][entity.value] = ''
        unique_matches = set()

        if self.relation_type != other.relation_type:
            return similarity_score, entity_matches

        for entity in self.entities:
            match, span, score = entity.get_most_similar_entity_LLM(other.entities, paper_text)
            if score >= similarity_threshold:
                entity_matches[entity.type][entity.value] = span
                unique_matches.add(match.id)

        similarity_score = len(unique_matches) / len(self.entities)

        return similarity_score, entity_matches

    def get_most_similar_relation(self, relations: List['Relation'], paper_text: str, entity_similarity_score: float):
        max_score = 0
        max_rel_id = None
        max_entity_matches = None
        for relation in relations:
            score, entity_matches = self.compute_similarity(relation, paper_text, entity_similarity_score)
            if score > max_score:
                max_score = score
                max_rel_id = relation.id
                max_entity_matches = entity_matches

        return max_rel_id, max_entity_matches

    def __str__(self):
        entities_rep_by_type = {}
        for entity in self.entities:
            if entity.type not in entities_rep_by_type:
                entities_rep_by_type[entity.type] = []
            entities_rep_by_type[entity.type].append(f'{entity.value}')

        return f'{json.dumps(entities_rep_by_type, indent=4)}-[{self.relation_type}]'


def get_default_metrics_dict(types: List[str], default_init: int) -> Dict:
    metrics = {}
    for t in types:
        metrics[t] = {'tp': default_init, 'fp': default_init, 'fn': default_init, 'total_pred_ent': default_init,
                      'pred_om': default_init, 'total_gold_ent': default_init, 'gold_om': default_init}

    return metrics


def get_doc_entity_agreement_metrics(prediction, gold, text, entity_types, logger):
    pred_match_by_type = {}
    for entity in prediction:
        if entity.type not in pred_match_by_type:
            pred_match_by_type[entity.type] = {}

        pred_match_by_type[entity.type][entity.id] = {'entity': entity, 'match': None}

    gold_match_by_type = {}
    for entity in gold:
        if entity.type not in gold_match_by_type:
            gold_match_by_type[entity.type] = {}

        gold_match_by_type[entity.type][entity.id] = {'entity': entity, 'match': None}

    log_str = '\n------TEXT------\n' + text + '\n------------------\n'

    for ent_type, pred_entities in pred_match_by_type.items():
        if ent_type not in gold_match_by_type:
            continue
        for pred_id, pred_match_info in pred_entities.items():
            pred_entity = pred_match_info['entity']
            gold_entities_of_type = [gold_match_info['entity'] for gold_match_info in
                                     gold_match_by_type[ent_type].values()]
            max_match, max_span, max_score = pred_entity.get_most_similar_entity_LLM(gold_entities_of_type, text)
            if max_match:
                log_str += f'PRED: {pred_entity.value} -> GOLD: {max_span}\n'
                pred_match_by_type[ent_type][pred_id]['match'] = max_match

    for ent_type, gold_entities in gold_match_by_type.items():
        if ent_type not in pred_match_by_type:
            continue
        for gold_id, gold_match_info in gold_entities.items():
            gold_entity = gold_match_info['entity']
            pred_entities_of_type = [pred_match_info['entity'] for pred_match_info in
                                     pred_match_by_type[ent_type].values()]
            max_match, max_span, max_score = gold_entity.get_most_similar_entity_LLM(pred_entities_of_type, text)
            if max_match:
                log_str += f'GOLD: {gold_entity.value}[{gold_entity.type}] -> PRED: {max_span}[{max_match.type}]\n'
                gold_match_by_type[ent_type][gold_id]['match'] = max_match

    ent_metrics = get_default_metrics_dict(entity_types, 0)

    for entity_type, pred_entities in pred_match_by_type.items():
        all_matches = [pred_match_info['match'].id for pred_match_info in pred_entities.values() if
                       pred_match_info['match']]
        unique_matches = set(all_matches)
        ent_metrics[entity_type]['tp'] += len(unique_matches)
        no_matches = [pred_match_info['entity'] for pred_match_info in pred_entities.values() if
                      not pred_match_info['match']]
        nr_no_matches = len(no_matches)
        ent_metrics[entity_type]['fp'] += nr_no_matches
        ent_metrics[entity_type]['pred_om'] += len(all_matches) - len(unique_matches)
        ent_metrics[entity_type]['total_pred_ent'] += len(pred_entities)

    for entity_type, gold_entities in gold_match_by_type.items():
        all_matches = [gold_match_info['match'].id for gold_match_info in gold_entities.values() if
                       gold_match_info['match']]
        unique_matches = set(all_matches)
        no_matches = [gold_match_info['entity'] for gold_match_info in gold_entities.values() if
                      not gold_match_info['match']]
        nr_no_matches = len(no_matches)
        ent_metrics[entity_type]['fn'] += nr_no_matches
        ent_metrics[entity_type]['gold_om'] += len(all_matches) - len(unique_matches)
        ent_metrics[entity_type]['total_gold_ent'] += len(gold_entities)

    if prediction or gold:
        logger.info(log_str)

    return ent_metrics


def compute_eval_criteria(metrics_by_type: Dict[str, Dict[str, float]], undefined_val=0) -> Dict[
    str, Dict[str, float]]:
    metrics = {}
    for t in metrics_by_type:
        tp, fp, fn = metrics_by_type[t]['tp'], metrics_by_type[t]['fp'], metrics_by_type[t]['fn']
        precision = tp / (tp + fp) if tp + fp > 0 else undefined_val
        recall = tp / (tp + fn) if tp + fn > 0 else undefined_val
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else undefined_val

        total_gold_ent = metrics_by_type[t]['total_gold_ent']
        gold_om = metrics_by_type[t]['gold_om']
        gold_over_matching = gold_om / total_gold_ent if total_gold_ent > 0 else undefined_val

        total_pred_ent = metrics_by_type[t]['total_pred_ent']
        pred_om = metrics_by_type[t]['pred_om']
        pred_over_matching = pred_om / total_pred_ent if total_pred_ent > 0 else undefined_val

        fp_ratio = fp / (tp + fp) if tp + fp > 0 else undefined_val

        metrics[t] = {'precision': precision, 'recall': recall, 'f1': f1, 'gold_over_matching': gold_over_matching,
                      'pred_over_matching': pred_over_matching, 'fp_ratio': fp_ratio, 'fp': fp, 'tp': tp, 'fn': fn}

    avg_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'gold_over_matching': 0, 'pred_over_matching': 0,
                   'fp_ratio': 0}
    for t in metrics:
        for key in avg_metrics:
            avg_metrics[key] += metrics[t][key]

    for key in avg_metrics:
        avg_metrics[key] /= len(metrics)

    metrics['avg'] = avg_metrics

    return metrics


def compute_entity_agreement(pred_entities: List, gold_entities: List, texts: List[str], entity_types: List[str],
                             logger, undefined_val=0):
    entity_metrics = get_default_metrics_dict(entity_types, 0)

    pbar = tqdm(total=len(pred_entities), desc='Computing entity agreement')
    for pred_entities_doc, gold_entities_doc, text in zip(pred_entities, gold_entities, texts):
        doc_ent_metrics = get_doc_entity_agreement_metrics(pred_entities_doc, gold_entities_doc, text, entity_types,
                                                           logger)

        for entity_type, metrics in doc_ent_metrics.items():
            for metric_name, metric_value in metrics.items():
                entity_metrics[entity_type][metric_name] += metric_value

        pbar.update(1)

    return compute_eval_criteria(entity_metrics, undefined_val)


def get_relations_match_table(relations_by_type) -> Dict:
    relations_match_table = {}
    for rel_type, relations in relations_by_type.items():
        if rel_type not in relations_match_table:
            relations_match_table[rel_type] = {}
        for relation in relations:
            matches = {}
            for entity_type, entities in relation.entities_by_type.items():
                if entity_type not in matches:
                    matches[entity_type] = {}
                for entity in entities:
                    matches[entity_type][entity.value] = ''

            relations_match_table[rel_type][relation.id] = {'relation': relation, 'matches': matches}

    return relations_match_table


def log_text(text: str):
    return f'-------------------<TEXT>-------------------\n{text}\n-------------------<\TEXT>-------------------\n\n'


def log_relations(rel_types: List[str], pred_match_by_type: Dict, gold_match_by_type: Dict):
    log_str = "-------------------<RELATIONS>-------------------\n"
    for rel_type in rel_types:
        log_str += f'----------------\n---{rel_type}---\n----------------\n'

        log_str += '---Pred-Relations---\n'
        if rel_type not in pred_match_by_type:
            log_str += f'* No pred relations of type {rel_type}\n'
        else:
            relations = pred_match_by_type[rel_type]
            for rel_id in relations:
                log_str += f'{relations[rel_id]["relation"]}\n'

        log_str += '---Gold-Relations---\n'
        if rel_type not in gold_match_by_type:
            log_str += f'* No gold relations of type {rel_type}\n'
        else:
            relations = gold_match_by_type[rel_type]
            for rel_id in relations:
                log_str += f'{relations[rel_id]["relation"]}\n'

    log_str += "-------------------<\RELATIONS>-------------------\n\n"

    return log_str


def log_entity_matches(entity_matches: Dict, header: str) -> str:
    log_str = f'-------------------<{header}>-------------------\n'
    for entity_type, entities in entity_matches.items():
        log_str += f'---{entity_type}---\n'
        for entity, match in entities.items():
            log_str += f'\'{entity}\' -> \'{match}\'\n'
        log_str += '---\n'
    log_str += f'-------------------<\{header}>-------------------\n\n'
    return log_str


def eval_rel_extraction_doc(prediction: Dict, gold: Dict, text: str, rel_types: List[str], ent_types: List[str], logger,
                            similarity_threshold):
    pred_match_by_type = get_relations_match_table(prediction)
    gold_match_by_type = get_relations_match_table(gold)
    log_str = "\n" + log_text(text)
    log_str += log_relations(rel_types, pred_match_by_type, gold_match_by_type)

    for rel_type, pred_relations in pred_match_by_type.items():
        if rel_type not in gold_match_by_type:
            continue

        gold_rels_of_type = [g['relation'] for g in gold_match_by_type[rel_type].values()]
        for pred_rel_id in pred_relations:
            pred_relation = pred_relations[pred_rel_id]['relation']
            gold_rel_id, entity_matches = pred_relation.get_most_similar_relation(gold_rels_of_type, text,
                                                                                  similarity_threshold)

            if gold_rel_id is not None:
                log_str += f'* Match for pred relation [{pred_rel_id}] is gold relation [{gold_rel_id}]\n'
                log_str += log_entity_matches(entity_matches, 'PRED-MATCHES')
                pred_match_by_type[rel_type][pred_rel_id]['matches'] = entity_matches

    for rel_type, gold_relations in gold_match_by_type.items():
        if rel_type not in pred_match_by_type:
            continue

        pred_rels_of_type = [p['relation'] for p in pred_match_by_type[rel_type].values()]
        for gold_rel_id in gold_relations:
            gold_rel = gold_relations[gold_rel_id]['relation']
            pred_rel_id, entity_matches = gold_rel.get_most_similar_relation(pred_rels_of_type, text,
                                                                             similarity_threshold)

            if pred_rel_id is not None:
                log_str += f'* Match for gold relation [{pred_rel_id}] is pred relation [{gold_rel_id}]\n'
                log_str += log_entity_matches(entity_matches, 'GOLD-MATCHES')
                gold_match_by_type[rel_type][gold_rel_id]['matches'] = entity_matches

    ent_metrics = get_default_metrics_dict(ent_types, 0)
    rel_metrics = get_default_metrics_dict(rel_types, 0)

    for rel_type, relations in pred_match_by_type.items():
        for rel_id in relations:
            unique_matches = set()
            unique_matches_by_type = {ent_type: set() for ent_type in ent_types}
            nr_matches = 0
            nr_matches_by_type = {ent_type: 0 for ent_type in ent_types}
            nr_no_matches = 0
            nr_no_matches_by_type = {ent_type: 0 for ent_type in ent_types}
            nr_entities = 0
            nr_entities_by_type = {ent_type: 0 for ent_type in ent_types}

            for ent_type, entity_matches in pred_match_by_type[rel_type][rel_id]['matches'].items():
                for entity, match in entity_matches.items():
                    if match:
                        unique_matches_by_type[ent_type].add(match)
                        unique_matches.add(match)
                        nr_matches_by_type[ent_type] += 1
                        nr_matches += 1
                    else:
                        nr_no_matches_by_type[ent_type] += 1
                        nr_no_matches += 1
                    nr_entities_by_type[ent_type] += 1
                    nr_entities += 1

            nr_unique_matches = len(unique_matches)
            nr_unique_matches_by_type = {ent_type: len(unique_matches_by_type[ent_type]) for ent_type in ent_types}

            rel_metrics[rel_type]['tp'] += nr_unique_matches
            rel_metrics[rel_type]['fp'] += nr_no_matches
            rel_metrics[rel_type]['total_pred_ent'] += nr_entities
            rel_metrics[rel_type]['pred_om'] += nr_matches - nr_unique_matches

            for ent_type in ent_types:
                ent_metrics[ent_type]['tp'] += nr_matches_by_type[ent_type]
                ent_metrics[ent_type]['fp'] += nr_no_matches_by_type[ent_type]
                ent_metrics[ent_type]['total_pred_ent'] += nr_entities_by_type[ent_type]
                ent_metrics[ent_type]['pred_om'] += nr_matches_by_type[ent_type] - nr_unique_matches_by_type[ent_type]

    for rel_type, relations in gold_match_by_type.items():
        for rel_id in relations:
            unique_matches = set()
            unique_matches_by_type = {ent_type: set() for ent_type in ent_types}
            nr_matches = 0
            nr_matches_by_type = {ent_type: 0 for ent_type in ent_types}
            nr_no_matches = 0
            nr_no_matches_by_type = {ent_type: 0 for ent_type in ent_types}
            nr_entities = 0
            nr_entities_by_type = {ent_type: 0 for ent_type in ent_types}

            for ent_type, entity_matches in gold_match_by_type[rel_type][rel_id]['matches'].items():
                for entity, match in entity_matches.items():
                    if match:
                        unique_matches_by_type[ent_type].add(match)
                        unique_matches.add(match)
                        nr_matches_by_type[ent_type] += 1
                        nr_matches += 1
                    else:
                        nr_no_matches_by_type[ent_type] += 1
                        nr_no_matches += 1
                    nr_entities_by_type[ent_type] += 1
                    nr_entities += 1

            nr_unique_matches = len(unique_matches)
            nr_unique_matches_by_type = {ent_type: len(unique_matches_by_type[ent_type]) for ent_type in ent_types}

            rel_metrics[rel_type]['fn'] += nr_no_matches
            rel_metrics[rel_type]['total_gold_ent'] += nr_entities
            rel_metrics[rel_type]['gold_om'] += nr_matches - nr_unique_matches

            for ent_type in ent_types:
                ent_metrics[ent_type]['fn'] += nr_no_matches_by_type[ent_type]
                ent_metrics[ent_type]['total_gold_ent'] += nr_entities_by_type[ent_type]
                ent_metrics[ent_type]['gold_om'] += nr_matches_by_type[ent_type] - nr_unique_matches_by_type[ent_type]

    if prediction or gold:
        logger.info(log_str)

    return rel_metrics, ent_metrics


def eval_rel_extraction(pred_rels, gold_rels, texts, rel_types, ent_types, logger, similarity_threshold):
    rel_metrics = get_default_metrics_dict(rel_types, 0)
    ent_metrics = get_default_metrics_dict(ent_types, 0)
    class_metrics = get_default_metrics_dict(['relevant', 'irrelevant'], 0)

    pbar = tqdm(total=len(pred_rels), desc='Evaluating relation extraction')
    for pred, gold, text in zip(pred_rels, gold_rels, texts):
        logger.info(f'---Evaluating document {pbar.n + 1}---')

        if pred and gold:
            class_metrics['relevant']['tp'] += 1
        elif not pred and not gold:
            class_metrics['irrelevant']['tp'] += 1
        elif pred and not gold:
            class_metrics['relevant']['fp'] += 1
            class_metrics['irrelevant']['fn'] += 1
        elif not pred and gold:
            class_metrics['relevant']['fn'] += 1
            class_metrics['irrelevant']['fp'] += 1

        doc_rel_metrics, doc_ent_metrics = eval_rel_extraction_doc(pred, gold, text, rel_types, ent_types, logger,
                                                                   similarity_threshold)
        for rel_type in rel_types:
            for key in doc_rel_metrics[rel_type]:
                rel_metrics[rel_type][key] += doc_rel_metrics[rel_type][key]

        for ent_type in ent_types:
            for key in doc_ent_metrics[ent_type]:
                ent_metrics[ent_type][key] += doc_ent_metrics[ent_type][key]
        pbar.update(1)

    return compute_eval_criteria(rel_metrics), compute_eval_criteria(ent_metrics), compute_eval_criteria(class_metrics)
