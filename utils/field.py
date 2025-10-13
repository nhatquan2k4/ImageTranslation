from torchtext.vocab import build_vocab_from_iterator

def build_vocab(tokenizer, texts):
    def yield_tokens():
        for text in texts:
            yield tokenizer.tokenize(text)

    vocab = build_vocab_from_iterator(
        yield_tokens(),
        specials=["<unk>", "<pad>", "<sos>", "<eos>"]
    )
    vocab.set_default_index(vocab["<unk>"])
    return vocab
