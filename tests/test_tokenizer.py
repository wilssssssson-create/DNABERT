import random

from dnabert_lite import KmerTokenizer, mask_tokens


def test_tokenize_and_encode_decode():
    tokenizer = KmerTokenizer(k=3)

    assert tokenizer.tokenize("ACGTA") == ["ACG", "CGT", "GTA"]

    ids = tokenizer.encode("ACGTA")
    assert ids[0] == tokenizer.cls_id
    assert tokenizer.decode(ids) == ["ACG", "CGT", "GTA"]


def test_mask_tokens_skips_special_tokens():
    tokenizer = KmerTokenizer(k=3)
    ids = tokenizer.encode("ACGTACGT")

    masked, labels = mask_tokens(
        ids,
        mask_id=tokenizer.mask_id,
        vocab_size=len(tokenizer),
        special_token_ids=tokenizer.special_ids,
        mask_probability=0.5,
        rng=random.Random(1),
    )

    assert labels[0] == -100
    assert masked[0] == tokenizer.cls_id
    assert any(label != -100 for label in labels[1:])
