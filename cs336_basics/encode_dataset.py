import argparse
import time

import numpy as np

from cs336_basics.tokenizer import Tokenizer


def encode_file_to_uint16(
    input_path: str,
    vocab_path: str,
    merges_path: str,
    special_tokens: list[str],
    output_path: str,
    chunk_size: int = 1_000_000,
) -> int:

    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)

    total_tokens = 0
    buffer: list[int] = []

    with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "wb") as f_out:
        for token_id in tokenizer.encode_iterable(f_in):
            buffer.append(token_id)
            if len(buffer) >= chunk_size:
                arr = np.array(buffer, dtype=np.uint16)
                # Sanity check: vocab must fit in uint16 (max 65535 entries)
                assert arr.max(initial=0) < 65536, "vocab_size exceeds uint16 range!"
                arr.tofile(f_out)
                total_tokens += len(buffer)
                buffer.clear()

        if buffer:
            arr = np.array(buffer, dtype=np.uint16)
            assert arr.max(initial=0) < 65536, "vocab_size exceeds uint16 range!"
            arr.tofile(f_out)
            total_tokens += len(buffer)

    return total_tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to input text file")
    parser.add_argument("--vocab", required=True, help="Path to vocab.json")
    parser.add_argument("--merges", required=True, help="Path to merges.txt")
    parser.add_argument(
        "--special-tokens",
        nargs="*",
        default=["<|endoftext|>"],
        help="Special tokens to preserve as single tokens",
    )
    parser.add_argument("--output", required=True, help="Path to output .bin file")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="Number of tokens to buffer before flushing to disk",
    )
    args = parser.parse_args()

    start = time.time()
    n_tokens = encode_file_to_uint16(
        input_path=args.input,
        vocab_path=args.vocab,
        merges_path=args.merges,
        special_tokens=args.special_tokens,
        output_path=args.output,
        chunk_size=args.chunk_size,
    )
    elapsed = time.time() - start

    print(f"Wrote {n_tokens:,} tokens to {args.output}")
    print(f"Took {elapsed:.1f}s ({n_tokens / max(elapsed, 1e-9):,.0f} tokens/sec)")


if __name__ == "__main__":
    main()
