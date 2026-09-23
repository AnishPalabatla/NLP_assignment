import json
import regex as re
import numpy as np
from typing import Iterable,Iterator


class Tokenizer:
    def __init__(self,vocab,merges,special_tokens=None):
        self.vocab=vocab
        self.merges=merges
        self.special_tokens=special_tokens

        # Add special tokens to vocab
        if self.special_tokens:
            for special_token in self.special_tokens:
                special_token_bytes=special_token.encode("utf-8")

                if special_token_bytes not in self.vocab.values():
                    self.vocab[len(self.vocab)]=special_token_bytes

        # Create byte->token ID lookup once
        self.vocab_ID={
            b:i
            for i,b in self.vocab.items()
        }

        # Store merge rank once.
        # Earlier merges have lower rank and should be applied first.
        self.merge_ranks={
            pair:i
            for i,pair in enumerate(self.merges)
        }

        # Pre-tokenization regex
        self.PAT=(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| """
            r"""?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

        self.pretoken_pattern=re.compile(self.PAT)

        # Prepare special-token regex once
        if self.special_tokens:
            self.sorted_special_tokens=sorted(
                self.special_tokens,
                key=len,
                reverse=True
            )

            self.special_pattern="|".join(
                re.escape(tok)
                for tok in self.sorted_special_tokens
            )

            self.special_regex=re.compile(
                f"({self.special_pattern})"
            )
        else:
            self.sorted_special_tokens=[]
            self.special_pattern=None
            self.special_regex=None

    @classmethod
    def from_files(
        cls,
        vocab_filepath,
        merges_filepath,
        special_tokens=None
    ):
        # Read vocab from file
        with open(vocab_filepath) as f1:
            raw_vocab=json.load(f1)

        vocab={
            int(v):bytes(k,"latin1")
            for k,v in raw_vocab.items()
        }

        # Read merges from file
        merges=[]

        with open(merges_filepath) as f2:
            for line in f2:
                line=line.rstrip()

                if line and not line.startswith("#"):
                    m=line.split(" ",-1)

                    if len(m)==2:
                        merges.append(
                            (
                                bytes(m[0],"latin1"),
                                bytes(m[1],"latin1")
                            )
                        )

        return cls(
            vocab,
            merges,
            special_tokens
        )

    def encode1(self,text:str)->list[int]:
        # Split on special tokens
        if self.special_tokens:
            splits=self.special_regex.split(text)
        else:
            splits=[text]

        IDs=[]

        for split in splits:
            if not split:
                continue

            # Directly add special token ID
            if (
                self.special_tokens
                and split in self.special_tokens
            ):
                special_bytes=split.encode("utf-8")
                IDs.append(self.vocab_ID[special_bytes])
                continue

            # Pre-tokenization
            for match in self.pretoken_pattern.finditer(split):
                pretoken=match.group(0)

                # Convert pre-token to bytes
                pretoken_bytes=[
                    bytes([b])
                    for b in pretoken.encode("utf-8")
                ]

                while len(pretoken_bytes)>1:

                    # Find the earliest-ranked merge that
                    # actually occurs in this pre-token.
                    best_pair=None
                    best_rank=float("inf")

                    for i in range(len(pretoken_bytes) - 1):
                        pair=(
                            pretoken_bytes[i],
                            pretoken_bytes[i+1]
                        )

                        rank=self.merge_ranks.get(pair)

                        if rank is not None and rank<best_rank:
                            best_rank=rank
                            best_pair=pair

                    if best_pair is None:
                        break

                    # Apply the selected merge
                    new_token=[]
                    i=0

                    while i<len(pretoken_bytes):
                        if (
                            i<len(pretoken_bytes) - 1
                            and (
                                pretoken_bytes[i],
                                pretoken_bytes[i+1]
                            )==best_pair
                        ):
                            new_token.append(
                                best_pair[0]+best_pair[1]
                            )
                            i += 2
                        else:
                            new_token.append(
                                pretoken_bytes[i]
                            )
                            i += 1

                    pretoken_bytes=new_token

                # Convert final tokens to IDs
                IDs.extend(
                    self.vocab_ID[b]
                    for b in pretoken_bytes
                )

        return IDs

    def encode(self,text:str)->list[int]:
        # Split on special tokens
        if self.special_tokens:
            splits=self.special_regex.split(text)
        else:
            splits=[text]

        IDs=[]

        for split in splits:
            if not split:
                continue

            # Directly add special token ID
            if (
                self.special_tokens
                and split in self.special_tokens
            ):
                special_bytes=split.encode("utf-8")
                IDs.append(self.vocab_ID[special_bytes])
                continue

            # Pre-tokenization
            for match in self.pretoken_pattern.finditer(split):
                pretoken=match.group(0)

                # Convert pre-token to bytes
                pretoken_bytes=[
                    bytes([b])
                    for b in pretoken.encode("utf-8")
                ]

                # -------------------------------------------------
                # Optimized BPE merging
                #
                # Instead of checking every merge in self.merges,
                # only inspect the pairs that actually occur in
                # this pre-token.
                # -------------------------------------------------
                while len(pretoken_bytes)>1:

                    best_pair=None
                    best_rank=float("inf")

                    # Find the highest-priority applicable merge
                    for i in range(len(pretoken_bytes) - 1):
                        pair=(
                            pretoken_bytes[i],
                            pretoken_bytes[i+1]
                        )

                        rank=self.merge_ranks.get(pair)

                        if rank is not None and rank<best_rank:
                            best_rank=rank
                            best_pair=pair

                    # No merge applies
                    if best_pair is None:
                        break

                    # Apply the selected merge to all occurrences
                    new_token=[]
                    i=0

                    while i<len(pretoken_bytes):

                        if (
                            i<len(pretoken_bytes) - 1
                            and (
                                pretoken_bytes[i],
                                pretoken_bytes[i+1]
                            )==best_pair
                        ):
                            new_token.append(
                                best_pair[0]+best_pair[1]
                            )
                            i += 2

                        else:
                            new_token.append(
                                pretoken_bytes[i]
                            )
                            i += 1

                    pretoken_bytes=new_token

                # Convert final token bytes to IDs
                IDs.extend(
                    self.vocab_ID[b]
                    for b in pretoken_bytes
                )

        return IDs

    def encode_iterable(
        self,
        iterable:Iterable[str]
    )->Iterator[int]:

        # Keep this streaming.
        # We do NOT read the entire iterable into memory.
        for each in iterable:
            yield from self.encode(each)

    def decode(self,ids:list[int])->str:
        # Concatenate all byte sequences
        byte_sequence=b"".join(
            self.vocab[ID]
            for ID in ids
        )

        # Decode UTF-8
        result=byte_sequence.decode(
            "utf-8",
            errors="replace"
        )

        return result


if __name__=="__main__":
    special_tokens=["<|endoftext|>"]

    tokenizer=Tokenizer.from_files(
        "../results/vocab.json",
        "../results/merges.txt",
        special_tokens
    )

    with open(
        "../data/TinyStoriesV2-GPT4-train.txt",
        "r",
        encoding="utf-8"
    ) as f:
        text=f.read()

    IDs=tokenizer.encode(text)

    np.save(
        "../results/tokens.npy",
        np.asarray(IDs)
    )