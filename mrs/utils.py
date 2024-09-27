import logging
import pandas as pd
from typing import List
from mrs.schemas import Session


class SessionBuilder:
    def __init__(self, style: str):
        self.style = style
        self.logger = self._set_logger()

    def _set_logger(self):
        logger = logging.getLogger(__name__)
        return logger

    def build_sessions(self, data_path: str) -> List[List[str]]:
        # data to load must be separated with `tab`
        data = pd.read_csv(data_path, sep="\t")
        styles = data.columns.tolist()
        if self.style not in styles:
            raise ValueError(f"Unsupported style. Style must be one of {styles}.\nInput: {self.style}")

        # use specified style conversational data
        data = data[[self.style]]
        data["group"] = data[self.style].isnull().cumsum()
        n_sessions = data["group"].iat[-1] + 1
        self.logger.debug(f"Number of sessions: {n_sessions}")

        # split data into sessions
        sessions: List[Session] = []
        groups = data.groupby("group", as_index=False, group_keys=False)

        for i, group in groups:
            session = group.dropna()[self.style].tolist()
            sessions += [Session(conv=session)]

        assert n_sessions == len(sessions)
        return sessions

    def build_short_sessions(self, sessions: List[Session], ctx_len: int = 4) -> List[Session]:
        short_sessions = []
        for session in sessions:
            for i in range(len(session.conv) - ctx_len + 1):
                short_sessions.append(Session(conv=session.conv[i : i + ctx_len]))
        return short_sessions

    def get_utterances(self, sessions: List[Session]):
        all_utts = set()
        for session in sessions:
            for utt in session.conv:
                all_utts.add(utt)
        return list(all_utts)
