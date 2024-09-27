import logging
import random
import torch
from transformers import AutoTokenizer
from collections import defaultdict
from typing import List
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from mrs.schemas import Session
from mrs.utils import SessionBuilder

MODEL_NAME = "klue/roberta-base"
DATA_ROOT = "../data/"
DATA_PATH = DATA_ROOT + "smilestyle_dataset.tsv"


class FinetuneDataset(Dataset):
    def __init__(self, builder: SessionBuilder, train: bool = True):
        # set logger
        self.logger = self._set_logger()

        # set tokenizer
        self.tokenizer = self._set_tokenizer()

        # build session_dataset
        sessions = builder.build_sessions(data_path=DATA_PATH)
        utts = builder.get_utterances(sessions)
        sessions = sessions[:-15] if train else sessions[-15:]
        self.session_dataset = self._set_session_dataset(utts, sessions)

    def _set_logger(self):
        logger = logging.getLogger(__name__)
        return logger

    def _set_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        special_tokens = {"sep_token": "<SEP>"}
        tokenizer.add_special_tokens(special_tokens)
        return tokenizer

    def _set_session_dataset(self, utts: List[str], sessions: List[Session], n_turns: int = 5, n_negs: int = 4) -> dict:
        data_json = defaultdict(dict)
        cnt = 0
        for session in sessions:
            ctx = [session.conv[0]]
            for turn in range(1, len(session.conv)):
                utt = session.conv[turn]
                neg_candidates = random.sample(utts, n_negs)
                data_json[cnt]["context"] = ctx[:][-n_turns:]
                data_json[cnt]["positive_response"] = utt
                data_json[cnt]["negative_responses"] = neg_candidates
                ctx.append(utt)
                cnt += 1
        return data_json

    def __len__(self):
        return len(self.session_dataset)

    def __getitem__(self, idx):
        session = self.session_dataset[idx]

        context = session["context"]
        positive_response = session["positive_response"]
        negative_responses = session["negative_responses"]
        session_tokens, session_labels = [], []

        # build context tokens
        context_tokens = [self.tokenizer.cls_token_id]
        for utt in context:
            context_tokens.extend(self.tokenizer.encode(utt, add_special_tokens=False))
            context_tokens.append(self.tokenizer.sep_token_id)

        # build positive response tokens
        positive_response_tokens = [self.tokenizer.eos_token_id]
        positive_response_tokens.extend(self.tokenizer.encode(positive_response, add_special_tokens=False))
        positive_response_tokens = context_tokens + positive_response_tokens
        session_tokens.append(positive_response_tokens)
        session_labels.append(1)

        # build negative response tokens
        for negative_response in negative_responses:
            negative_response_tokens = [self.tokenizer.eos_token_id]
            negative_response_tokens.extend(self.tokenizer.encode(negative_response, add_special_tokens=False))
            negative_response_tokens = context_tokens + negative_response_tokens
            session_tokens.append(negative_response_tokens)
            session_labels.append(0)

        return_value = {"session_tokens": session_tokens, "session_labels": session_labels}
        return return_value


class FinetuneDatasetCollator:
    def __init__(self, pad_idx: int, max_length: int):
        self.pad_idx = pad_idx
        self.max_length = max_length

    def __call__(self, batch: List[dict]):
        # |batch| = [
        # {
        #   'session_tokens': [[session1_tensor], [session2_tensor], ...]
        #   'session_labels': [session1_label, session2_label, ...]
        # }, ...
        # ]
        batch_input_tokens, batch_input_labels = [], []
        for sample in batch:
            for session in sample["session_tokens"]:
                batch_input_tokens.append(torch.tensor(session))
            batch_input_labels.extend(sample["session_labels"])

        batch_input_tokens = pad_sequence(batch_input_tokens, batch_first=True, padding_value=self.pad_idx)
        batch_input_attentions = (batch_input_tokens != self.pad_idx).long()
        batch_labels = torch.tensor(batch_input_labels).long()

        return {
            "batch_input_tokens": batch_input_tokens,
            "batch_input_attentions": batch_input_attentions,
            "batch_labels": batch_labels,
        }
