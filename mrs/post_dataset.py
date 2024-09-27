import logging
import numpy as np
import torch
import random
from typing import List, Dict

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from mrs.schemas import Session
from mrs.utils import SessionBuilder

MODEL_NAME = "klue/roberta-base"
DATA_ROOT = "../data/"
DATA_PATH = DATA_ROOT + "smilestyle_dataset.tsv"


class PostDataset(Dataset):
    def __init__(self, builder: SessionBuilder):
        # set logger
        self.logger = self._set_logger()

        # set tokenizer
        self.tokenizer = self._set_tokenizer()

        # build sessions
        self.sessions = builder.build_sessions(data_path=DATA_PATH)
        self.short_sessions = builder.build_short_sessions(self.sessions, ctx_len=4)
        self.utts = builder.get_utterances(self.sessions)

    def _set_logger(self):
        logger = logging.getLogger(__name__)
        return logger

    def _set_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        special_tokens = {"sep_token": "<SEP>"}
        tokenizer.add_special_tokens(special_tokens)
        return tokenizer

    def _mask_tokens(self, tokens: List, ratio: float = 0.15) -> List:
        tokens = np.array(tokens)
        n_mask = int(len(tokens) * ratio)
        mask_pos = random.sample(range(len(tokens)), n_mask)

        # fancy indexing
        tokens[mask_pos] = self.tokenizer.mask_token_id
        tokens = tokens.tolist()
        return tokens

    def _get_mask_positions(self, tokens: List) -> List:
        tokens = np.array(tokens)
        mask_positions = np.where(tokens == self.tokenizer.mask_token_id)[0].tolist()
        return mask_positions

    def construct_mlm_inputs(self, short_session: Session) -> Dict[str, list]:
        corrupt_tokens, output_tokens = [], []

        for i, utt in enumerate(short_session.conv):
            tokens = self.tokenizer.encode(utt, add_special_tokens=False)
            masked_tokens = self._mask_tokens(tokens)

            if i == len(short_session.conv) - 1:
                output_tokens.extend(tokens)
                corrupt_tokens.extend(masked_tokens)
            else:
                output_tokens.extend(tokens + [self.tokenizer.sep_token_id])
                corrupt_tokens.extend(masked_tokens + [self.tokenizer.sep_token_id])

        corrupt_mask_positions = self._get_mask_positions(corrupt_tokens)
        return_value = {
            "output_tokens": output_tokens,
            "corrupt_tokens": corrupt_tokens,
            "corrupt_mask_positions": corrupt_mask_positions,
        }

        return return_value

    def construct_urc_inputs(self, short_session: Session) -> Dict[str, List]:
        urc_tokens, ctx_utts = [], []

        for i in range(len(short_session.conv)):
            utt = short_session.conv[i]
            tokens = self.tokenizer.encode(utt, add_special_tokens=False)

            if i == len(short_session.conv) - 1:
                urc_tokens += [self.tokenizer.eos_token_id]
                positive_tokens = [self.tokenizer.cls_token_id] + urc_tokens + tokens

                while True:
                    random_neg_response = random.choice(self.utts)
                    if random_neg_response not in ctx_utts:
                        break
                random_neg_response_token = self.tokenizer.encode(random_neg_response, add_special_tokens=False)
                random_tokens = [self.tokenizer.cls_token_id] + urc_tokens + random_neg_response_token
                ctx_neg_response = random.choice(ctx_utts)
                ctx_neg_response_token = self.tokenizer.encode(ctx_neg_response, add_special_tokens=False)
                ctx_neg_tokens = [self.tokenizer.cls_token_id] + urc_tokens + ctx_neg_response_token
            else:
                urc_tokens += tokens + [self.tokenizer.sep_token_id]

            ctx_utts.append(utt)

        return_value = {
            "positive_tokens": positive_tokens,
            "random_negative_tokens": random_tokens,
            "context_negative_tokens": ctx_neg_tokens,
            "urc_labels": [0, 1, 2],
        }
        return return_value

    def __len__(self):
        return len(self.short_sessions)

    def __getitem__(self, idx):
        # ---- input data for MLM ---- #
        short_session = self.short_sessions[idx]
        mlm_input = self.construct_mlm_inputs(short_session)

        # ---- intput data for utterance relevance classification ---- #
        urc_input = self.construct_urc_inputs(short_session)

        return_value = dict()
        return_value["mlm_input"] = mlm_input
        return_value["urc_input"] = urc_input

        return return_value


class PostDatasetCollator:
    def __init__(self, pad_idx: int, max_length: int):
        self.pad_idx = pad_idx
        self.max_length = max_length

    def __call__(self, batch: List[dict]):
        # |batch| = [
        # {
        # 'mlm_input': {
        # 'output_tokens': list(),
        # 'corrupt_tokens': list(),
        # 'corrupt_mask_positions': list()
        # },
        # 'urc_input': {
        # 'positive_tokens': list(),
        # 'random_negative_tokens': list(),
        # 'context_negative_tokens': list(),
        # 'urc_labels': list()
        # }
        # }, ...
        # ]

        # initialize batch output bags
        mlm_output_tokens_inputs, mlm_corrupt_tokens_inputs = [], []
        urc_inputs = []
        mlm_corrupt_mask_positions, urc_labels = [], []

        for sample in batch:
            mlm_input, urc_input = sample["mlm_input"], sample["urc_input"]

            mlm_output_tokens_inputs.append(torch.tensor(mlm_input["output_tokens"][: self.max_length]))
            mlm_corrupt_tokens_inputs.append(torch.tensor(mlm_input["corrupt_tokens"][: self.max_length]))
            mlm_corrupt_mask_positions.append(torch.tensor(mlm_input["corrupt_mask_positions"]))

            urc_inputs.append(torch.tensor(urc_input["positive_tokens"][: self.max_length]))
            urc_inputs.append(torch.tensor(urc_input["random_negative_tokens"][: self.max_length]))
            urc_inputs.append(torch.tensor(urc_input["context_negative_tokens"][: self.max_length]))
            urc_labels.append(torch.tensor(urc_input["urc_labels"]))

        # pad sequence
        mlm_output_tokens_inputs = pad_sequence(mlm_output_tokens_inputs, batch_first=True, padding_value=self.pad_idx)
        mlm_corrupt_tokens_inputs = pad_sequence(
            mlm_corrupt_tokens_inputs, batch_first=True, padding_value=self.pad_idx
        )

        urc_inputs = pad_sequence(urc_inputs, batch_first=True, padding_value=self.pad_idx)

        # get attention masking positions
        mlm_attentions = (mlm_output_tokens_inputs != self.pad_idx).long()
        urc_attentions = (urc_inputs != self.pad_idx).long()

        return_value = {
            "mlm_inputs": {
                "output_tokens": mlm_output_tokens_inputs,
                "corrupt_tokens": mlm_corrupt_tokens_inputs,
                "mask_positions": mlm_corrupt_mask_positions,
                "attention_masks": mlm_attentions,
            },
            "urc_inputs": {
                "input_tokens": urc_inputs,
                "labels": urc_labels,
                "attention_masks": urc_attentions,
            },
        }
        return return_value
