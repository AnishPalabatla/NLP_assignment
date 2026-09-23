from cs336_basics.pretokenization_example import find_chunk_boundaries
from collections import Counter
import regex as re
import multiprocessing as mp

# Parallelize pretoken counting using multiprocessing
def process_chunk(args):
    input_path,start,end,special_tokens=args

    # read file by chunk
    with open(input_path,"rb") as f:
        f.seek(start)
        chunk=f.read(end-start).decode("utf-8",errors="ignore")

    # Split the chunk into pre-tokens based on special tokens
    splits=re.split("|".join(re.escape(tok) for tok in special_tokens),chunk)

    # further split by regex
    pretokens=[]
    for split in splits:
        PAT=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pretokens+=[n.group(0) for n in re.finditer(PAT,split)]

    pretoken_counts=Counter(pretokens)

    byte_counts={tuple(bytes([b]) for b in token.encode("utf-8")):count for token,count in pretoken_counts.items()}

    return byte_counts

def train_bpe(
    input_path:str,
    vocab_size:int,
    special_tokens:list[str]
) -> tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:
    num_processes=mp.cpu_count()
    with open(input_path,"rb") as f:
        boundaries=find_chunk_boundaries(f,num_processes,"<|endoftext|>".encode("utf-8"))

    chunk_args=[(input_path,start,end,special_tokens) for start,end in zip(boundaries[:-1],boundaries[1:])]
    with mp.Pool(processes=num_processes) as pool:
        results=pool.map(process_chunk,chunk_args)

    all_pretoken_counts=Counter()
    for c in results:
        all_pretoken_counts.update(c)

    pairs_counts=Counter()
    pair_to_pretokens={}

    for pretoken,count in all_pretoken_counts.items():
        if len(pretoken)<2:
            continue
        seen_here=set()
        for i in range(len(pretoken)-1):
            pair=(pretoken[i],pretoken[i+1])
            pairs_counts[pair]+=count
            seen_here.add(pair)
        for pair in seen_here:
            pair_to_pretokens.setdefault(pair,set()).add(pretoken)

    vocab=[tok.encode("utf-8") for tok in special_tokens]+[bytes([i]) for i in range(256)]
    merges=[]

    while len(vocab)<vocab_size:
        if not pairs_counts:
            break

        most_common_pair=max(pairs_counts.items(),key=lambda x:(x[1],x[0]))[0]
        merges.append(most_common_pair)
        new_token=most_common_pair[0]+most_common_pair[1]
        vocab.append(new_token)

        affected=pair_to_pretokens.pop(most_common_pair,set())

        for pretoken in affected:
            count=all_pretoken_counts.pop(pretoken,None)
            if count is None:
                continue

            for i in range(len(pretoken)-1):
                pair=(pretoken[i],pretoken[i+1])
                pairs_counts[pair] -= count
                if pairs_counts[pair]<=0:
                    del pairs_counts[pair]
                pair_to_pretokens.get(pair,set()).discard(pretoken)

            new_pretoken=[]
            i=0
            while i<len(pretoken):
                if i<len(pretoken)-1 and (pretoken[i],pretoken[i+1]) == most_common_pair:
                    new_pretoken.append(new_token)
                    i+=2
                else:
                    new_pretoken.append(pretoken[i])
                    i+=1
            new_pretoken=tuple(new_pretoken)

            all_pretoken_counts[new_pretoken]=all_pretoken_counts.get(new_pretoken,0)+count

            for i in range(len(new_pretoken)-1):
                pair=(new_pretoken[i],new_pretoken[i+1])
                pairs_counts[pair]+=count
                pair_to_pretokens.setdefault(pair,set()).add(new_pretoken)

    vocab_mapping={i:v for i,v in enumerate(vocab)}
    return vocab_mapping,merges