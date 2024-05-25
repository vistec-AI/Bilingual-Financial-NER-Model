import os
import json
import argparse
from datetime import datetime

class ArgumentParserUtility:
    @staticmethod
    def parse_arguments():
        parser = argparse.ArgumentParser(description="Named Entity Prompt Generator")
        parser.add_argument("--scheme", type=str, default="IOBES", help="Scheme name")
        parser.add_argument("--sent_length", type=int, default=512, help="Sentence length")
        parser.add_argument("--data_path_train", type=str, default="data/train.txt", help="Dataset name")
        parser.add_argument("--data_path_dev", type=str, default="data/dev.txt", help="Dataset name")
        parser.add_argument("--data_path_test", type=str, default="data/test.txt", help="Dataset name")
        parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
        parser.add_argument("--pretrained", type=str, default="bert-base-uncased", help="Pretrained model name")
        parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
        parser.add_argument("--epochs", type=int, default=100, help="Epochs")
        parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
        parser.add_argument("--early_stopping", type=int, default=8, help="Early stoping")
        parser.add_argument("--eval_per_epoch", type=int, default=1, help="Evaluate per epoch")
        parser.add_argument("--name", type=str, default="", help="Project name")
        parser.add_argument("--mode", type=str, default="train", help="Mode")
        parser.add_argument("--seed", type=int, default=42, help="Random seed")
        parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/base", help="Checkpoint dir")
        parser.add_argument("--resume", type=str, default=None, help="Resume")
        args = parser.parse_args()
        return args

    @staticmethod
    def setup_path(args):
        unique_id = str(datetime.now().strftime("%Y%m%d_%H%M%S%f"))
        args.checkpoint_dir = os.path.join(args.checkpoint_dir, unique_id)
        path_config = args.checkpoint_dir + "/config.json"
        return path_config

    @staticmethod
    def save_args_to_json(args, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as file:
            json.dump(vars(args), file, indent=4)

    @staticmethod
    def load_args_from_json(filename):
        with open(filename, 'r') as file:
            data = json.load(file)
        return argparse.Namespace(**data)