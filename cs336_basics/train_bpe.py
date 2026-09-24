from cs336_basics.pretokenization_example import find_chunk_boundaries
from collections import Counter
import regex as re
import multiprocessing as mp
import json
import os
import time

# Resolve paths relative to this script's location, not the current working
# directory -- avoids "file not found" errors when running from a different folder.
SCRIPT_DIR=os.path.dirname(os.path.abspath(__file__))
DATA_PATH=os.path.join(SCRIPT_DIR, "..", "data", "TinyStoriesV2-GPT4-train.txt")
RESULTS_DIR=os.path.join(SCRIPT_DIR, "..", "results")


# Parallelize pretoken counting using multiprocessing
def process_chunk(args):
    input_path, start, end, special_tokens=args

    # Read file by chunk
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk=f.read(end-start).decode("utf-8", errors="ignore")

    # Split the chunk into pre-tokens based on special tokens
    splits=re.split(
        "|".join(re.escape(tok) for tok in special_tokens),
        chunk
    )

    # Further split by regex
    pretokens=[]
    PAT=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    for split in splits:
        pretokens+=[
            n.group(0)
            for n in re.finditer(PAT, split)
        ]

    # Count occurrences of each pre-token
    pretoken_counts=Counter(pretokens)

    # Convert pre-tokens to byte tuples
    byte_counts={
        tuple(bytes([b]) for b in token.encode("utf-8")):count
        for token, count in pretoken_counts.items()
    }

    return byte_counts


def train_bpe(
    input_path:str,
    vocab_size:int,
    special_tokens:list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    num_processes=min(mp.cpu_count(),8)
    print(f"[train_bpe] cpu_count={mp.cpu_count()}, using {num_processes} processes", flush=True)

    t0=time.time()
    with open(input_path, "rb") as f:
        boundaries=find_chunk_boundaries(
            f,
            num_processes,
            "<|endoftext|>".encode("utf-8")
        )
    print(f"[train_bpe] chunk boundaries computed in {time.time()-t0:.1f}s", flush=True)

    # Multiprocessing to process chunks
    chunk_args=[
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    print("[train_bpe] starting pre-tokenization (pool.map)...", flush=True)
    t0=time.time()
    with mp.Pool(processes=num_processes) as pool:
        results=pool.map(process_chunk, chunk_args)
    print(f"[train_bpe] pre-tokenization done in {time.time()-t0:.1f}s", flush=True)

    # Combine counts from all processes
    all_pretoken_counts=Counter()
    for c in results:
        all_pretoken_counts.update(c)
    print(f"[train_bpe] {len(all_pretoken_counts)} unique pretokens", flush=True)

    # Initialize vocabulary
    vocab={
        i:bytes([i])
        for i in range(256)
    }

    for token in special_tokens:
        vocab[len(vocab)]=token.encode("utf-8")

    merges=[]

    # ---------------------------------------------------------
    # Initialize pair counts ONCE, and also build an index from
    # each pair -> the set of pretokens containing it. This lets
    # each merge step touch ONLY the affected pretokens instead
    # of rescanning the entire corpus every iteration (the old
    # implementation's real bottleneck:O(vocab_size*num_unique
    # _pretokens) instead of O(vocab_size*avg_affected)).
    # ---------------------------------------------------------
    pairs_counts=Counter()
    pair_to_pretokens:dict[tuple[bytes, bytes], set]={}

    for pretoken, count in all_pretoken_counts.items():
        if len(pretoken)<2:
            continue

        seen_pairs_here=set()
        for i in range(len(pretoken)-1):
            pair=(pretoken[i], pretoken[i+1])
            pairs_counts[pair]+=count
            seen_pairs_here.add(pair)

        for pair in seen_pairs_here:
            pair_to_pretokens.setdefault(pair, set()).add(pretoken)

    # ---------------------------------------------------------
    # Merge pairs until vocabulary size is reached
    # ---------------------------------------------------------
    merge_num=0
    t0=time.time()
    target_merges=vocab_size-len(vocab)

    while len(vocab)<vocab_size:

        # Break if there are no pairs left
        if not pairs_counts:
            break

        # Find the most common pair.
        # Tie-breaking:highest frequency first, then lexicographically largest pair.
        most_common_pair=max(
            pairs_counts,
            key=lambda x:(pairs_counts[x], x)
        )

        merges.append(most_common_pair)

        # Create the new token
        new_token=most_common_pair[0]+most_common_pair[1]
        vocab[len(vocab)]=new_token

        # Only the pretokens known to contain this pair need updating.
        affected=pair_to_pretokens.pop(most_common_pair, set())

        for pretoken in affected:
            count=all_pretoken_counts.pop(pretoken, None)
            if count is None:
                continue

            # -------------------------------------------------
            # Remove this pretoken's old pair contributions from
            # the global counts / index.
            # -------------------------------------------------
            for i in range(len(pretoken)-1):
                pair=(pretoken[i], pretoken[i+1])
                pairs_counts[pair]-=count
                if pairs_counts[pair] <= 0:
                    del pairs_counts[pair]
                pair_to_pretokens.get(pair, set()).discard(pretoken)

            # -------------------------------------------------
            # Apply a single left-to-right merge pass to this
            # pretoken (correct behavior even with repeated pairs,
            # e.g. "aaa" -> pair ('a','a') occurring twice).
            # -------------------------------------------------
            new_pretoken=[]
            i=0
            while i<len(pretoken):
                if i<len(pretoken)-1 and (pretoken[i], pretoken[i+1])==most_common_pair:
                    new_pretoken.append(new_token)
                    i+=2
                else:
                    new_pretoken.append(pretoken[i])
                    i+=1
            new_pretoken=tuple(new_pretoken)

            # Combine counts if multiple old pretokens collapse to the same new one.
            all_pretoken_counts[new_pretoken]=all_pretoken_counts.get(new_pretoken, 0)+count

            # -------------------------------------------------
            # Add the new pretoken's pair contributions.
            # -------------------------------------------------
            for i in range(len(new_pretoken)-1):
                pair=(new_pretoken[i], new_pretoken[i+1])
                pairs_counts[pair]+=count
                pair_to_pretokens.setdefault(pair, set()).add(new_pretoken)

        merge_num+=1
        if merge_num%200==0 or merge_num==target_merges:
            elapsed=time.time()-t0
            print(
                f"[train_bpe] merge {merge_num}/{target_merges}, "
                f"vocab_size={len(vocab)}, elapsed={elapsed:.1f}s",
                flush=True
            )

    print(f"[train_bpe] merging complete:{merge_num} merges in {time.time()-t0:.1f}s", flush=True)

    return vocab, merges


if __name__=="__main__":

    special_tokens=["<|endoftext|>"]

    print(f"[train_bpe] data path:{DATA_PATH}", flush=True)
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Training data not found at {DATA_PATH}")

    vocab, merges=train_bpe(DATA_PATH, 10000, special_tokens)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    vocab_path=os.path.join(RESULTS_DIR, "vocab.json")
    merges_path=os.path.join(RESULTS_DIR, "merges.txt")

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(
            {v.decode("latin1"):k for k, v in vocab.items()},
            f,
            ensure_ascii=False,
            indent=2
        )

    with open(merges_path, "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{a.decode('latin1')} {b.decode('latin1')}\n")

    print(f"[train_bpe] wrote {vocab_path} and {merges_path}", flush=True)
