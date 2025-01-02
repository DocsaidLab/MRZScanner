import re
from enum import IntEnum
from typing import Dict, List, Optional, Union

import numpy as np
from capybara import EnumCheckMixin


def replace_digits(text: str):
    text = re.sub('0', 'O', text)
    text = re.sub('1', 'I', text)
    text = re.sub('2', 'Z', text)
    text = re.sub('4', 'A', text)
    text = re.sub('5', 'S', text)
    text = re.sub('8', 'B', text)
    return text


def replace_letters(text: str):
    text = re.sub('O', '0', text)
    text = re.sub('Q', '0', text)
    text = re.sub('U', '0', text)
    text = re.sub('D', '0', text)
    text = re.sub('I', '1', text)
    text = re.sub('Z', '2', text)
    text = re.sub('A', '4', text)
    text = re.sub('S', '5', text)
    text = re.sub('B', '8', text)
    return text


def replace_sex(text: str):
    text = re.sub('P', 'F', text)
    text = re.sub('N', 'M', text)
    return text


class DecodeMode(EnumCheckMixin, IntEnum):
    Default = 0
    CTC = 1
    Normal = 2


class TextDecoder:

    def __init__(
        self,
        *,
        chars_dict: Dict[str, int],
        decode_mode: Optional[Union[DecodeMode, str, int]] = DecodeMode.Default
    ):
        self.chars_dict = chars_dict
        self.chars = {v: k for k, v in self.chars_dict.items()}
        self.decode_mode = DecodeMode.obj_to_enum(decode_mode)

    def decode(self, encode: List[np.ndarray]) -> List[List[str]]:
        encode = np.array(encode)
        if self.decode_mode == DecodeMode.CTC:
            masks = (encode != np.roll(encode, 1)) & (encode != 0)
        elif self.decode_mode in [DecodeMode.Default, DecodeMode.Normal]:
            masks = []
            for row in encode:
                eos_index = np.where(row == self.chars_dict["<EOS>"])[0]
                if eos_index.size > 0:
                    mask = np.zeros_like(row, dtype=bool)
                    mask[:eos_index[0]] = True
                else:
                    mask = np.ones_like(row, dtype=bool)
                mask = mask & (row != self.chars_dict["<PAD>"])
                masks.append(mask)

        chars_list = [''.join([self.chars[idx] for idx in e[m]])
                      for e, m in zip(encode, masks)]

        return chars_list

    def __call__(self, *args, **kwargs) -> List[List[str]]:
        return self.decode(*args, **kwargs)
