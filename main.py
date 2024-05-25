import os
import torch
import random
import evaluate
import numpy as np
from transformers import AutoTokenizer
from src.learner import Learner
from src.dataloader import TextProcessor
from src.dataloader import LabelProcessor
from src.dataloader import CoNLLDataset
from src.opts import ArgumentParserUtility
from src.model import TokenClassificationModel

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ## Data ##
    label_processor = LabelProcessor()
    do_lower_case = True if 'uncased' in args.pretrained else False
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained, do_lower_case=do_lower_case)
    text_processor = TextProcessor(tokenizer, max_length=args.sent_length)
    
    ## Model ##
    temp = args.data_path_train if args.mode=='train' else args.data_path_test
    tags_path = os.path.dirname(temp) + "/tags.txt"
    print(f"Tags path: {tags_path}")

    ## Load tags ##
    l2id = CoNLLDataset.load_tags(tags_path)
    model = TokenClassificationModel(
        args.pretrained, id2l={v: k for k, v in l2id.items()}, 
        ne_scheme=args.scheme, dropout=args.dropout)
    model.to(device)

    metric_name = "./seqeval" if os.path.exists('./seqeval') else 'seqeval'
    metric = evaluate.load(metric_name, cache_dir=args.checkpoint_dir)

    learner = Learner(args, model, metric, device)
    learner.load_checkpoint(args.resume)

    if args.mode == "train":
        train_dataset = CoNLLDataset(
            args.data_path_train, 
            tags_path, args.scheme, text_processor, 
            label_processor, filter_empty=True
        )

        dev_dataset = CoNLLDataset(
            args.data_path_dev, 
            tags_path, args.scheme, text_processor, 
            label_processor, filter_empty=True
        )
        
        learner.train(
            train_dataset=train_dataset,
            dev_dataset=dev_dataset
        )

    if args.mode in ["test", "train"]:
        test_dataset = CoNLLDataset(
            args.data_path_test, 
            tags_path, args.scheme, 
            text_processor, label_processor)
        
        f1, (score, report, results) = learner.evaluate(
            test_dataset, save_path=args.checkpoint_dir
        )
    else:
        print("Invalid mode")


if __name__ == "__main__":
    args = ArgumentParserUtility.parse_arguments()
    path_config = ArgumentParserUtility.setup_path(args)
    ArgumentParserUtility.save_args_to_json(args, path_config)
    args = ArgumentParserUtility.load_args_from_json(path_config)
    args.resume = None if args.resume == "None" else args.resume
    main(args)
