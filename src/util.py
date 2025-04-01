import json
import os
from typing import Dict, Tuple, List
import logging
import nltk
import pandas as pd
from datetime import datetime
from colorama import Fore, Style
from openai import OpenAI



NER_ENTITY_TYPES_ATTRIBUTES = [
    {
        'entity_type': 'comb-element',
        'desc': "An idea, method, model, technique, or approach combined in the text with other elements.",
        'nr_entities_in_example': 2
    },
    {
        'entity_type': 'analogy-src',
        'prompt_type_name': 'inspiration-src',
        'desc': "A concept, idea, problem, approach, or domain the authors drew inspiration from.",
        'nr_entities_in_example': 1
    },
    {
        'entity_type': 'analogy-target',
        'prompt_type_name': 'inspiration-target',
        'desc': "A concept, idea, problem, approach, or domain in which the authors utilize the inspiration they drew from the inspiration source.",
        'nr_entities_in_example': 2
    }
]

def map_chars_into_words(text_words: List[str], text: str) -> Dict[int, int]:
    char_to_word_index = {}
    current_char_index = 0

    text = text.replace("``", '"').replace("''", '"')
    for i, word in enumerate(text_words):
        current_char_index = text.find(word, current_char_index)
        if current_char_index == -1:
            raise ValueError(
                f"Tokenization error: Unable to find [word={word}]\n"
                f"in the text:\n{text}.\n"
                f"Text words are:\n{text_words}\n"
            )
        char_to_word_index[current_char_index] = i
        current_char_index += len(word)
    return char_to_word_index

def word_tokenize_text(text: str) -> List[str]:
    word_tokens = nltk.word_tokenize(text)
    # Undoing the tokenization of opening and closing quotes
    word_tokens = [word.replace("``", '"').replace("''", '"') for word in word_tokens]
    return word_tokens


def create_out_dir(output_dir: str) -> None:
    """Create an output directory if it does not exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


class ColoredFormatter(logging.Formatter):
    """Custom formatter to color log levels in the console."""

    COLORS = {
        "DEBUG": Fore.CYAN,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.MAGENTA + Style.BRIGHT
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, "")
        reset = Style.RESET_ALL
        record.levelname = f"{log_color}{record.levelname}{reset}"  # Color the level name
        return super().format(record)


def setup_default_logger(output_dir: str) -> logging.Logger:
    """Set up a logger that writes to output_dir with colored console output"""
    logs_dir = os.path.join(output_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all messages

    log_format = "[%(levelname)s] [%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"

    # Create formatters
    file_formatter = logging.Formatter(log_format)  # Standard formatter for files
    colored_formatter = ColoredFormatter(log_format)  # Colored formatter for console

    # Console Handler (prints to terminal with colors)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(colored_formatter)
    stream_handler.setLevel(logging.DEBUG)

    # File Handler (writes logs to a file)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = os.path.join(logs_dir, f'{timestamp}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)

    # Attach handlers
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    logger.info(f'Writing logs to {log_file}')
    return logger


def init_openai_client():
    key_path = 'openai_key'
    if not os.path.exists(key_path):
        raise ValueError('OpenAI key file not found')

    open_ai_key = open(key_path).read().strip()
    client = OpenAI(api_key=open_ai_key)
    return client


def request_openai_batch_completions(prompts: Dict[str, str], max_tokens: int, temperature: float, batch_idx: int,
                                     output_path: str, client, engine: str) -> str:
    batch_requests = []
    for prompt_id, prompt in prompts.items():
        batch_entry = {"custom_id": prompt_id, "method": "POST",
                       "url": "/v1/chat/completions",
                       "body": {"model": engine, "max_tokens": max_tokens, "temperature": temperature,
                                "messages": [{"role": "user", "content": prompt}]}}
        batch_requests.append(batch_entry)

    batch_requests = pd.DataFrame(batch_requests)
    batch_requests_file = os.path.join(output_path, f'batch_{batch_idx}_requests.json')
    batch_requests.to_json(batch_requests_file, lines=True, orient='records')

    batch_input_file = client.files.create(
        file=open(batch_requests_file, "rb"),
        purpose="batch"
    )

    batch_out = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": f'Batch {batch_idx}',
        }
    )

    return batch_out.id


def get_openai_batch_completions(batch_id: str, client) -> Tuple[Dict[str, str], str]:
    batch_status = client.batches.retrieve(batch_id)
    query_responses_by_id = {}
    if batch_status.status == "completed":
        batch_response = client.files.content(batch_status.output_file_id).text
        query_responses = [json.loads(r) for r in batch_response.strip().split('\n')]
        query_responses_by_id = {}
        for response in query_responses:
            response_content = response['response']['body']['choices'][0]['message']['content']
            query_responses_by_id[response['custom_id']] = response_content
    elif batch_status.status == "failed":
        raise Exception(f'Batch {batch_id} failed')

    return query_responses_by_id, batch_status.status
