import argparse
import os

from lputil import get_entities_data, get_eval_samples, MODELS, build_eval_dataset, build_train_dataset, \
    RankingEvaluator, load_model, get_model_prompts_dict
from datasets import Dataset
from datetime import datetime
from util import setup_default_logger

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses, \
    SentenceTransformerModelCardData

from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.training_args import SentenceTransformerTrainingArguments


def hpo_search_space(trial):
    return {
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 5),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0, 0.3),
        "learning_rate": trial.suggest_float("learning_rate", 2e-6, 2e-4, log=True),
    }


def run_hpo_training(train_data):
    eval_data = build_eval_dataset(args.valid_path, args.model_name)
    entity_id_to_text = get_entities_data(args.entities_path)
    samples, all_entity_ids = get_eval_samples(args.all_edges_path, args.valid_candidates_path, eval_data,
                                               entity_id_to_text)
    evaluator = RankingEvaluator(samples, all_entity_ids, entity_id_to_text, args.model_name, logger,
                                 args.encode_batch_size, args.output_path)
    eval_dataset = Dataset.from_pandas(eval_data)
    eval_dataset = eval_dataset.select_columns(['query', 'answer', 'label'])

    def hpo_model_init(trial):
        return load_model(args.weights_precision, args.quantize, args.checkpoint, args.model_name)

    def hpo_loss_init(model):
        return losses.ContrastiveLoss(model)

    def hpo_compute_objective(metrics):
        return metrics["eval_all_median_gold_rank"]

    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_path,
        bf16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        per_device_train_batch_size=args.batch_size,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=100,
        run_name="hpo"
    )

    prompts = get_model_prompts_dict(args.model_name)
    if prompts:
        training_args.prompts = prompts

    trainer = SentenceTransformerTrainer(
        model=None,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_dataset,
        evaluator=evaluator,
        model_init=hpo_model_init,
        loss=hpo_loss_init,
    )

    best_trial = trainer.hyperparameter_search(
        hp_space=hpo_search_space,
        compute_objective=hpo_compute_objective,
        n_trials=args.hpo_trials,
        direction="minimize",
        backend="optuna",
    )
    logger.info(best_trial)

    return best_trial


def main():
    model = load_model(args.weights_precision, args.quantize, args.checkpoint, args.model_name)
    eval_data = build_eval_dataset(args.test_path, args.model_name)
    entities_data = get_entities_data(args.entities_path)
    eval_samples, all_entities_ids = get_eval_samples(args.all_edges_path, args.test_candidates_path, eval_data,
                                                      entities_data)

    evaluator = RankingEvaluator(eval_samples, all_entities_ids, entities_data, args.model_name,
                                 logger, args.encode_batch_size, args.output_path)

    if args.zero_shot:
        evaluator(model)
        return

    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_path,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_steps=100,
        save_strategy='epoch',
        save_total_limit=1,
    )

    prompts = get_model_prompts_dict(args.model_name)
    if prompts:
        training_args.prompts = prompts

    train_data = build_train_dataset(args.all_edges_path, args.train_path, args.model_name, args.nr_negatives)
    train_dataset = train_data.select_columns(['query', 'answer', 'label'])
    logger.info(f'Created training dataset with {len(train_data)} examples')

    if args.do_hpo:
        best_trial = run_hpo_training(train_dataset)
        logger.info(f'Best trial: {best_trial}')
        for key, value in best_trial.hyperparameters.items():
            setattr(training_args, key, value)

    loss = losses.ContrastiveLoss(model)

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
    )

    trainer.train()
    checkpoint_path = os.path.join(args.output_path, 'model')
    model.save_pretrained(checkpoint_path, safe_serialization=True)
    evaluator(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--test_path', type=str)
    parser.add_argument('--valid_path', type=str)
    parser.add_argument('--entities_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--nr_negatives', type=int)
    parser.add_argument('--all_edges_path', type=str)
    parser.add_argument('--test_candidates_path', type=str)
    parser.add_argument('--valid_candidates_path', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--zero_shot', action='store_true')
    parser.add_argument('--encode_batch_size', type=int, default=256)
    parser.add_argument('--eval_candidates_cutoff_year', type=int, default=None)
    parser.add_argument('--quantize', action='store_true')
    parser.add_argument('--weights_precision', type=int, default=16)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--do_hpo', action='store_true')
    parser.add_argument('--hpo_trials', type=int, default=2)

    args = parser.parse_args()

    run_name = args.model_name.split('/')[-1]
    if args.zero_shot:
        run_name += '_zero_shot'

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    args.output_path = os.path.join(args.output_path, f'{run_name}_{timestamp}')

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    logger = setup_default_logger(args.output_path)

    main()
