import argparse
import sentencepiece as spm


def train(input_txt, model_prefix, vocab_size=32000, model_type='bpe'):
    args = f"--input={input_txt} --model_prefix={model_prefix} --vocab_size={vocab_size} --model_type={model_type} --character_coverage=1.0 --unk_id=0 --bos_id=1 --eos_id=2 --pad_id=3"
    spm.SentencePieceTrainer.Train(args)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--model_prefix', default='bpe_vi')
    p.add_argument('--vocab_size', default=32000, type=int)
    p.add_argument('--model_type', default='bpe')
    args = p.parse_args()
    train(args.input, args.model_prefix, args.vocab_size, args.model_type)