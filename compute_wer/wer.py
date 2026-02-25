# Copyright (c) 2025, Zhendong Peng (pzd17@tsinghua.org.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import List, Optional, Set
from unicodedata import east_asian_width

from edit_distance import SequenceMatcher


class WER:
    def __init__(
        self,
        reference: Optional[List[str]] = None,
        hypothesis: Optional[List[str]] = None,
        hotwords: Optional[Set[str]] = None, # 新增参数
    ):
        self.equal = 0
        self.replace = 0
        self.delete = 0
        self.insert = 0

        if reference is not None and hypothesis is not None:
            self.reference = []
            self.hypothesis = []
            self.tokens = defaultdict(WER)

            # --- 逻辑A: 预标记热词索引 (基于字符或词序列) ---
            hot_indices = set()
            if hotwords:
                for hw in hotwords:
                    hw = hw.strip()
                    if not hw: continue
                    
                    # === 核心修改开始：中英文混合切分逻辑 ===
                    hw_tokens = []
                    # 1. 先按空格把字符串炸开（保证英文单词分开，也能处理带空格的中文）
                    raw_parts = hw.split()
                    
                    for part in raw_parts:
                        # 如果全是 ASCII 字符 (英文/数字/半角符号)，保持为一个整体 (Token)
                        if all(ord(c) < 128 for c in part):
                            hw_tokens.append(part)
                        else:
                            # 3. 如果包含中文 (非ASCII)，则暴力拆解成单字
                            hw_tokens.extend(list(part))

                    # 4. 滑动窗口匹配 (逻辑不变)
                    n = len(hw_tokens)
                    if n == 0: continue
                    # 在 reference 列表中寻找这一串 token
                    for i in range(len(reference) - n + 1):
                        if reference[i : i + n] == hw_tokens:
                            for k in range(n):
                                hot_indices.add(i + k)
                        
            matcher = SequenceMatcher(reference, hypothesis)
            for op, i, i_end, j, j_end in matcher.get_opcodes():
                
                # 确定当前操作的权重
                weight = 1
                if op != "insert":
                    # 当前 reference 索引处于热词范围内，权重为 10
                    if i in hot_indices:
                        weight = 10
                else:
                    # 针对插入错误：仅当插入点紧接在热词之后，认为受热词影响，权重为 10
                    # 注意：不对"热词之前的插入"加权，否则句首热词会导致所有句首插入都被错误地赋高权重
                    if i > 0 and (i - 1) in hot_indices:
                        weight = 10

                # 应用权重：修改对象本身的属性
                setattr(self, op, getattr(self, op) + weight)
                
                # 统计每个 token 具体的错误（用于汇总）
                token = reference[i] if op != "insert" else hypothesis[j]
                if token not in self.tokens:
                    self.tokens[token] = WER()
                self.tokens[token][op] += weight

                # 可视化文本对齐 (保持原样)
                ref_token = reference[i] if op != "insert" else ""
                hyp_token = hypothesis[j] if op != "delete" else ""
                diff = WER.width(hyp_token) - WER.width(ref_token)
                self.reference.append(ref_token + " " * diff)
                self.hypothesis.append(hyp_token + " " * -diff)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    @staticmethod
    def width(token: str) -> int:
        """
        Get the width of a token.

        Args:
            token: The token to get the width.
        Returns:
            The width of the token.
        """
        return sum(1 + (east_asian_width(char) in "AFW") for char in token)

    @property
    def all(self) -> int:
        return self.equal + self.replace + self.delete + self.insert

    @property
    def wer(self) -> float:
        if self.all == 0:
            return 0
        return (self.replace + self.delete + self.insert) / self.all

    def __str__(self) -> str:
        return f"{self.wer * 100:4.2f} % N={self.all} Cor={self.equal} Sub={self.replace} Del={self.delete} Ins={self.insert}"

    def update(self, other: "WER"):
        """
        Update this WER with another WER.

        Args:
            other (WER): The other WER.
        """
        self.equal += other.equal
        self.replace += other.replace
        self.delete += other.delete
        self.insert += other.insert

    @staticmethod
    def overall(wers: List["WER"]) -> "WER":
        """
        Calculate the overall WER.

        Args:
            wers: The list of WERs.
        Returns:
            The overall WER.
        """
        overall = WER()
        for wer in wers:
            if wer is None:
                continue
            for key in ("equal", "replace", "delete", "insert"):
                overall[key] += wer[key]
        return overall


class SER:
    def __init__(self):
        self.cor = 0
        self.err = 0

    @property
    def all(self) -> int:
        return self.cor + self.err

    @property
    def ser(self) -> float:
        return self.err / self.all if self.all != 0 else 0

    def __str__(self) -> str:
        return f"{self.ser * 100:4.2f} % N={self.all} Cor={self.cor} Err={self.err}"
