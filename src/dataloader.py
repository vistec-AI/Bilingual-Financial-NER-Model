import torch
from torch.utils.data import Dataset


class TextProcessor:
    def __init__(self, tokenizer, max_length: int):
        self.tok = tokenizer
        self.max_len = max_length
        self.bos, self.eos, self.pad = self.tok.encode(
            '', add_special_tokens=True, max_length=3, padding='max_length')
    def process(self, tokens):
        enc_dict, ids = {}, [self.bos]
        shift = 1
        for idx, word in enumerate(tokens):
            word_ids = self.tok.encode(word, add_special_tokens=False)
            ids.extend(word_ids)
            enc_dict[idx] = (shift, shift + len(word_ids))
            shift += len(word_ids)

        ids.append(self.eos)
        ids.extend([self.pad] * (self.max_len - len(ids)))
        mask = [1 if i != self.pad else 0 for i in ids]

        if len(ids) > self.max_len:
            print("Warning: The number of tokens exceeds max_length. Some tokens will be truncated.")
            breakpoint()

        return {
            'input_text': tokens, 
            'input_ids': ids, 
            'attention_mask': mask, 
            'encode_dict': enc_dict
        }


class LabelProcessor:
    def __init__(self, corrector=None, convertor=None):
        self.corrector = corrector
        self.convertor = convertor

    def encode(self, original_labels, encode_dict, output_length):
        entities = self.bioes_to_entities(original_labels)
        shfited_entities = self.shifting(entities, encode_dict)
        adapted_conll = self.entities_to_bioes(shfited_entities, output_length)
        return adapted_conll
    
    def shifting(self, entities, encode_dict):
        shfited_labels = []
        for index in range(len(entities)):
            start, end, tag = entities[index]
            start = encode_dict[start][0]
            end = encode_dict[end-1][-1]
            shfited_labels.append((start, end, tag))
        return shfited_labels
    
    def bioes_to_entities(self, label_list):
        entities = []
        start_pos = None
        entity_type = None

        for idx, label in enumerate(label_list):
            if label.startswith('B-'):
                if start_pos is not None:
                    entities.append((start_pos, idx, entity_type))
                start_pos = idx
                entity_type = label.split('B-')[-1]
            elif label.startswith('I-'):
                if start_pos is None:
                    start_pos = idx
                entity_type = label.split('I-')[-1]
            elif label.startswith('S-'):
                entities.append((idx, idx + 1, label.split('S-')[-1]))
            elif label.startswith('E-'):
                if start_pos is not None:
                    entities.append((start_pos, idx + 1, entity_type))
                    start_pos = None
                else:
                    entities.append((idx, idx + 1, label.split('E-')[-1]))
            else:
                if start_pos is not None:
                    entities.append((start_pos, idx, entity_type))
                    start_pos = None
        if start_pos is not None:
            entities.append((start_pos, len(label_list), entity_type))
        return entities
    
    def entities_to_bioes(self, entities, max_seq_length):
        labels = ['O'] * max_seq_length
        for start, end, label_type in entities:
            if end > max_seq_length or start < 0 or start > end:
                raise ValueError(f"Invalid entity boundaries or order: {(start, end, label_type)}")
            if start == end - 1:
                labels[start] = f'S-{label_type}'
            else:
                labels[start] = f'B-{label_type}'
                for i in range(start + 1, end - 1):
                    labels[i] = f'I-{label_type}'
                labels[end - 1] = f'E-{label_type}'
        return labels
    
    @staticmethod
    def display(org_bioes, encode_dict, entities, shfited_entities):
        print("\n\nOrginal BIOES: ", org_bioes)
        print("\nEncode dictionary (original):")
        for ids in range(len(org_bioes)):
            print(f"{ids}: {encode_dict[ids]}\t{org_bioes[ids]}") 

        print("\n Entities (original) to (shifted):")
        for ids in range(len(shfited_entities)):
            print(f"{ids}: {entities[ids]}\t{shfited_entities[ids]}")
    

def split_token_label(temp):
    tokens, labels = [], [] 
    for line in temp:
        line = line.split()
        word, label = ['_', line[-1]] if len(line)==1 else line
        tokens.append(word)
        labels.append(label)
    assert len(tokens) == len(labels)
    return tokens, labels


def load_conll_dataset_sub_doc_level(file_path):
    with open(file_path, 'r') as f:
        data = f.readlines()
    documents = []
    current_document = []
    for line in data:
        if len(line.split())==0:
            if len(current_document) > 0:
                documents.append(current_document)
            current_document = []
        else:
            if line.startswith('START_DOC:'): continue
            current_document.append(line)
    if len(current_document) > 0:
        documents.append(current_document)
    return documents


class CoNLLDataset(Dataset):
    def __init__(self, file_path, tags_path, scheme, text_processor, label_processor, filter_empty=False):
        self.scheme = scheme
        self.text_processor = text_processor
        self.label_processor = label_processor
        self.filename = file_path.split('/')[-1]

        self.l2id = self.load_tags(tags_path)
        self.id2l = {v: k for k, v in self.l2id.items()}

        scheme = "O" + scheme.replace("O", "")
        self.s2id = {v: k for k, v in enumerate(scheme)}
        self.id2s = {v: k for k, v in self.s2id.items()}

        data = load_conll_dataset_sub_doc_level(file_path)
        self.data = [split_token_label(d) for d in data]
        print("MAX_LEN", max([len(x[0]) for x in self.data]))
        self.data = [item for item in self.data if len(item[0]) > 0]
        
        if filter_empty:
            ### Filter out sentences with only 'O' labels
            self.data = [item for item in self.data if any([label != 'O' for label in item[1]])]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, debug=False):
        sentence, org_labels = self.data[idx]
        assert len(sentence) == len(org_labels)
        item = self.text_processor.process(sentence)
        labels = self.label_processor.encode(org_labels, item['encode_dict'], len(item['input_ids']))
        item['type_ids'] = [self.l2id.get(x, 0) for x in labels]
        span_tag = [x.split('-')[0] for x in labels]
        item['span_ids'] = [self.s2id.get(x) for x in span_tag]
        
        if debug:
            print(f"Orginal labels: {org_labels}\nAdapted labels: {labels}")
        return item

    @staticmethod
    def load_tags(tags_path):
        with open(tags_path, "r") as f:
            labels = [l.strip() for l in f]
        labels.remove('O')
        return {label: i for i, label in enumerate(['O'] + labels)}


def data_collector(batch):
    keys = ['input_ids', 'attention_mask', 'span_ids', 'type_ids']
    batched_data = {key: [item[key] for item in batch] for key in keys}
    
    for key, values in batched_data.items():
        batched_data[key] = torch.stack([torch.tensor(val) for val in values])

    return batched_data
