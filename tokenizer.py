# from pathlib import Path
from tokenizers.implementations import ByteLevelBPETokenizer

# paths = [str(x) for x in Path("./").glob("**/*.txt")]
paths = ["./datasets/eng_news-typical_2016_1M-sentences.txt"]
is_train = False
if is_train:
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=paths, vocab_size=50_000, min_frequency=2, special_tokens=[
        "<s>",
        "</s>",
        "<pad>",
        "<unk>",
        "<mask>",
    ])

    tokenizer.save_model("./EsperBerto")
else:
    from tokenizers.processors import BertProcessing
    tokenizer = ByteLevelBPETokenizer(
        vocab="./EsperBerto/vocab.json",
        merges="./EsperBerto/merges.txt"
    )
    tokenizer.post_processor = BertProcessing(
        ("</s>",tokenizer.token_to_id("</s>")),
        ("<s>",tokenizer.token_to_id("<s>")),
    )
    tokenizer.enable_truncation(max_length=512)
    e = tokenizer.encode("hello, you are right!")
    print(e.tokens)

