from cs336_basics.pretokenization_example import find_chunk_boundaries
from collections import Counter
import regex as re
import multiprocessing as mp
import json
import os


# Parallelize pretoken counting using multiprocessing
def process_chunk(args):
    input_path, start, end, special_tokens=args

    # Read file by chunk
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk=f.read(end - start).decode("utf-8", errors="ignore")

    # Split the chunk into pre-tokens based on special tokens
    splits=re.split(
        "|".join(re.escape(tok) for tok in special_tokens),
        chunk
    )

    # Further split by regex
    pretokens=[]
    PAT=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    for split in splits:
        pretokens += [
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

    num_processes=min(mp.cpu_count(), 4)

    with open(input_path, "rb") as f:
        boundaries=find_chunk_boundaries(
            f,
            num_processes,
            "<|endoftext|>".encode("utf-8")
        )

    # Multiprocessing to process chunks
    chunk_args=[
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    with mp.Pool(processes=num_processes) as pool:
        results=pool.map(process_chunk, chunk_args)

    # Combine counts from all processes
    all_pretoken_counts=Counter()

    for c in results:
        all_pretoken_counts.update(c)

    # Initialize vocabulary
    vocab={
        i:bytes([i])
        for i in range(256)
    }

    for token in special_tokens:
        vocab[len(vocab)]=token.encode("utf-8")

    merges=[]

    # ---------------------------------------------------------
    # Initialize pair counts ONCE.
    #
    # The old implementation recalculated every pair from
    # every pre-token on every merge. That was the main
    # performance bottleneck.
    # ---------------------------------------------------------
    pairs_counts=Counter()

    for pretoken, count in all_pretoken_counts.items():
        if len(pretoken)<2:
            continue

        for i in range(len(pretoken) - 1):
            pair=(pretoken[i], pretoken[i+1])
            pairs_counts[pair] += count

    # ---------------------------------------------------------
    # Merge pairs until vocabulary size is reached
    # ---------------------------------------------------------
    while len(vocab)<vocab_size:

        # Break if there are no pairs left
        if not pairs_counts:
            break

        # Find the most common pair.
        #
        # This preserves the original tie-breaking behavior:
        # highest frequency first, then lexicographically
        # largest pair.
        most_common_pair=max(
            pairs_counts,
            key=lambda x:(pairs_counts[x], x)
        )

        merges.append(most_common_pair)

        # Create the new token
        new_token=(
            most_common_pair[0]
           +most_common_pair[1]
        )

        vocab[len(vocab)]=new_token

        new_pretoken_counts={}

        # -----------------------------------------------------
        # Update only the pair counts affected by this merge.
        # -----------------------------------------------------
        for pretoken, count in all_pretoken_counts.items():

            if len(pretoken)<2:
                new_pretoken_counts[pretoken]=(
                    new_pretoken_counts.get(pretoken, 0)
                   +count
                )
                continue

            # Check whether this pre-token contains the pair
            contains_pair=False

            for i in range(len(pretoken) - 1):
                if (
                    pretoken[i],
                    pretoken[i+1]
                )==most_common_pair:
                    contains_pair=True
                    break

            # If this pre-token is unaffected, keep it unchanged
            if not contains_pair:
                new_pretoken_counts[pretoken]=(
                    new_pretoken_counts.get(pretoken, 0)
                   +count
                )
                continue

            # -------------------------------------------------
            # Remove the old pair contributions from the global
            # pair counts.
            # -------------------------------------------------
            for i in range(len(pretoken) - 1):
                pair=(
                    pretoken[i],
                    pretoken[i+1]
                )

                pairs_counts[pair] -= count

                if pairs_counts[pair] <= 0:
                    del pairs_counts[pair]

            # -------------------------------------------------
            # Apply the merge to this pre-token.
            # -------------------------------------------------
            new_pretoken=[]

            i=0

            while i<len(pretoken):

                if (
                    i<len(pretoken) - 1
                    and (
                        pretoken[i],
                        pretoken[i+1]
                    )==most_common_pair
                ):
                    new_pretoken.append(new_token)
                    i += 2

                else:
                    new_pretoken.append(pretoken[i])
                    i += 1

            new_pretoken=tuple(new_pretoken)

            # Combine counts if multiple old pre-tokens become
            # the same new pre-token.
            new_pretoken_counts[new_pretoken]=(
                new_pretoken_counts.get(new_pretoken, 0)
               +count
            )

            # -------------------------------------------------
            # Add the new pair contributions.
            # -------------------------------------------------
            for i in range(len(new_pretoken) - 1):
                pair=(
                    new_pretoken[i],
                    new_pretoken[i+1]
                )

                pairs_counts[pair] += count

        # Replace the old pre-token dictionary
        all_pretoken_counts=new_pretoken_counts

    return vocab, merges


if __name__=="__main__":

    special_tokens=["<|endoftext|>"]

    vocab, merges=train_bpe(
        "data/TinyStoriesV2-GPT4-train.txt",
        10000,
        special_tokens
    )

    os.makedirs("results", exist_ok=True)

    with open(
        "results/vocab.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            {
                v.decode("latin1"):k
                for k, v in vocab.items()
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    with open(
        "results/merges.txt",
        "w",
        encoding="utf-8"
    ) as f:
        for a, b in merges:
            f.write(
                f"{a.decode('latin1')} "
                f"{b.decode('latin1')}\n"
            )