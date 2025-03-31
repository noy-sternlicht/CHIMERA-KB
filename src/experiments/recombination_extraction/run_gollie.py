import argparse
import json
import os

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from typing import List
import inspect
from src.tasks.utils_typing import Template, dataclass
from jinja2 import Template as jinja2Template

"""
Relation definitions
"""


@dataclass
class Inspiration(Template):
    """An inspiration describes drawing inspiration or similarities from one concept, idea, problem,
    approach, or domain and implementing it in another. For example, taking inspiration from the human brain to
    design a learning algorithm, performing a reduction from one problem to another, or using a technique from one
    domain in another."""

    inspiration_src: str  # The source of the inspiration (e.g., the human brain)
    inspiration_target: str  # The target of the inspiration (e.g., a learning algorithm)


@dataclass
class Combination(Template):
    """A combination describes joining two ideas, methods, models, techniques to obtain a certain goal.  For example,
    combining two models to improve performance, combining two methods to solve a problem, or combining two ideas to
    create a new concept."""

    comb_element_1: str  # The first element of the combination (e.g., model A)
    comb_element_2: str  # The second element of the combination (e.g., model B)


ENTITY_DEFINITIONS: List[Template] = [
    Inspiration,
    Combination,
]

template_txt = (
    """# The following lines describe the task definition
{%- for definition in guidelines %}
{{ definition }}
{%- endfor %}

# This is the text to analyze
text = {{ text.__repr__() }}

# The annotation instances that take place in the text above are listed here
result = [
{%- for ann in annotations %}
    {{ ann }},
{%- endfor %}
]""")


def main():
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True,
                                                 torch_dtype=torch.bfloat16).to("cuda")

    guidelines = [inspect.getsource(definition) for definition in ENTITY_DEFINITIONS]
    template = jinja2Template(template_txt)
    test_data = pd.read_csv(args.eval_path, dtype={'paper_id': str})

    pred_relations = {}
    for _, example in test_data.iterrows():
        formated_text = template.render(guidelines=guidelines, text=example['text'])
        prompt, _ = formated_text.split("result =")
        prompt = prompt + "result ="

        model_input = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
        model_input["input_ids"] = model_input["input_ids"][:, :-1]
        model_input["attention_mask"] = model_input["attention_mask"][:, :-1]

        model_ouput = model.generate(
            **model_input.to(model.device),
            max_new_tokens=128,
            do_sample=False,
            min_new_tokens=0,
            num_beams=1,
            num_return_sequences=1,
        )[0]

        relations = {}
        response = tokenizer.decode(model_ouput, skip_special_tokens=True).split("result = ")[-1]
        response = response.strip()
        response = response[1:-1].strip()
        if not response:
            pred_relations[example['paper_id']] = relations
            continue
        response = response.replace('\n', '').strip().split("),")
        for res in response:
            res = res.strip()
            if not res:
                continue
            res_type = res.split("(")[0].strip()
            if res_type not in ['Combination', 'Inspiration']:
                print(f"Unknown relation type {res_type}")
                continue
            res_type = 'combination' if res_type == 'Combination' else 'inspiration'
            if len(res.split("(")) < 2:
                print(f"Invalid relation format: {res}")
                continue
            res_entities = res.split("(")[1].strip().split(",")
            res_entities = [e for e in res_entities if e]
            if len(res_entities) < 2:
                print(f"Less than two entities in the relation: {res_entities}")
                continue

            if res_type == 'combination':
                res_entities = {'comb-element': [x.split("=")[1].strip()[1:-1] for x in res_entities]}
            else:
                res_entities = {'inspiration-src': [res_entities[0].split("inspiration_src")[1].strip()[1:-1]],
                                'inspiration-target': [res_entities[1].split("inspiration_target")[1].strip()[1:-1]]}

            if res_type not in relations:
                relations[res_type] = []
            relations[res_type].append(res_entities)

        pred_relations[example['paper_id']] = relations
        print(f"Predicted relations for {example['paper_id']}: {relations}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    processed_pred_relations = {}
    for paper_id, relations in pred_relations.items():
        processed_relations = []
        combination_relations = relations.get('combination', [])
        processed_combination_relations = []
        for relation in combination_relations:
            found_rel = False
            for rel_set in processed_combination_relations:
                entities = relation['comb-element'] if 'comb-element' in relation else []
                for ent in entities:
                    if ent in rel_set:
                        rel_set.update(entities)
                        found_rel = True
                        break
                if found_rel:
                    break
            if not found_rel:
                processed_combination_relations.append(set(relation['comb-element']))

        comb_entities = list(processed_combination_relations[0]) if processed_combination_relations else []
        if comb_entities:
            combination = {'type': 'combination', 'entities': {'comb-element': comb_entities}}
            processed_relations.append(combination)
            print(f"Processed relations for {paper_id}: {combination}")

        inspiration_relations = relations.get('inspiration', [])
        if inspiration_relations:
            inspiration = {'type': 'inspiration', 'entities': inspiration_relations[0]}
            processed_relations.append(inspiration)
            print(f"Inspiration relations for {paper_id}: {inspiration}")

        if len(processed_relations) > 1:
            print(f"Multiple relations found for {paper_id}: {processed_relations}")

        processed_pred_relations[paper_id] = processed_relations

    out_file = os.path.join(args.output_dir, "predictions.json")
    with open(out_file, 'w') as f:
        json.dump(pred_relations, f)

    print(f"Predictions saved to {out_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--eval_path', type=str)
    parser.add_argument('--model_name', type=str, default="HiTZ/GoLLIE-7B")

    args = parser.parse_args()
    main()
