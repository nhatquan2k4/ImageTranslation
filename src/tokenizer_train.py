import argparse
import sentencepiece as spm
from pathlib import Path


def train(input_txt: str, model_prefix: str, vocab_size: int = 32000, model_type: str = 'bpe'):
    input_path = Path(input_txt)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_txt}")
    
    args_str = (
        f"--input={input_txt} --model_prefix={model_prefix} --vocab_size={vocab_size} "
        f"--model_type={model_type} --character_coverage=1.0 --unk_id=0 --bos_id=1 --eos_id=2 --pad_id=3 "
        f"--pad_id=3 --user_defined_symbols=__SEP__"  # Thêm nếu cần special tokens
    )
    spm.SentencePieceTrainer.Train(args_str)
    print(f"✅ Tokenizer trained: {model_prefix}.model")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Train SentencePiece Tokenizer")
    p.add_argument('--input', required=True, help="Path to input text file")
    p.add_argument('--model_prefix', default='bpe_vi', help="Prefix for model files")
    p.add_argument('--vocab_size', default=32000, type=int)
    p.add_argument('--model_type', default='bpe', choices=['bpe', 'unigram'])
    args = p.parse_args()
    train(args.input, args.model_prefix, args.vocab_size, args.model_type)