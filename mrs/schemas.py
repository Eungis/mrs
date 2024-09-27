from typing import List
from dataclasses import dataclass


@dataclass
class Session:
    conv: List[str]
    """Conversation: List of utterences"""
