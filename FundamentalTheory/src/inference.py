# inference.py
import argparse
import torch
from PIL import Image
import sentencepiece as spm
from torchvision import transforms
from model import ViTEncoder, TransformerDecoderModel, Image2TextModel
from typing import List, Tuple


def beam_search_decode(
    model: Image2TextModel,
    img: torch.Tensor,
    sp: spm.SentencePieceProcessor,
    beam_width: int = 4,
    max_len: int = 128,
    device: str = "cpu",
    pad_idx: int = 3,
    bos_id: int = 1,
    eos_id: int = 2
) -> str:
    model.eval()
    with torch.no_grad():
        # Encoder forward
        memory = model.encoder(img.unsqueeze(0).to(device))  # (1, S, d_model)

        # Initialize beam: list of (sequence_ids, score)
        beams: List[Tuple[List[int], float]] = [([bos_id], 0.0)]  # Start with BOS

        for _ in range(max_len - 1):
            new_beams = []
            for seq_ids, score in beams:
                if seq_ids[-1] == eos_id:
                    new_beams.append((seq_ids, score))  # Keep finished beams
                    continue

                # Prepare input for decoder
                ys = torch.tensor([seq_ids], dtype=torch.long, device=device)  # (1, T)
                T = ys.size(1)
                tgt_mask = (torch.triu(torch.ones(T, T, device=device)) == 1).transpose(0, 1)
                tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))

                # Decoder forward
                out = model.decoder(ys, memory, tgt_mask=tgt_mask)
                out = out.transpose(0, 1)  # (1, T, vocab)
                prob = out[:, -1, :]  # (1, vocab) - probs for next token
                log_probs = torch.log_softmax(prob, dim=-1).squeeze(0)  # (vocab,)

                # Get top-k candidates
                top_probs, top_ids = torch.topk(log_probs, beam_width, dim=-1)
                for i in range(beam_width):
                    new_seq = seq_ids + [top_ids[i].item()]
                    new_score = score + top_probs[i].item()
                    new_beams.append((new_seq, new_score))

            # Select top beam_width beams
            new_beams.sort(key=lambda x: x[1], reverse=True)  # Descending score
            beams = new_beams[:beam_width]

            # Early stop if all beams finished
            if all(seq[-1] == eos_id for seq, _ in beams):
                break

        # Select best beam
        best_seq, _ = beams[0]
        # Decode only until EOS
        eos_idx = best_seq.index(eos_id) if eos_id in best_seq else len(best_seq)
        return sp.decode_ids(best_seq[:eos_idx + 1])


def greedy_decode(
    model: Image2TextModel,
    img: torch.Tensor,
    sp: spm.SentencePieceProcessor,
    max_len: int = 128,
    device: str = "cpu",
    pad_idx: int = 3,
    bos_id: int = 1,
    eos_id: int = 2
) -> str:
    model.eval()
    with torch.no_grad():
        memory = model.encoder(img.unsqueeze(0).to(device))

        ys = torch.tensor([[bos_id]], dtype=torch.long, device=device)
        for i in range(max_len - 1):
            T = ys.size(1)
            tgt_mask = (torch.triu(torch.ones(T, T, device=device)) == 1).transpose(0, 1)
            tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))

            out = model.decoder(ys, memory, tgt_mask=tgt_mask)
            out = out.transpose(0, 1)
            prob = out[:, -1, :]
            next_word = torch.argmax(prob, dim=-1).item()

            ys = torch.cat([ys, torch.tensor([[next_word]], device=device)], dim=1)
            if next_word == eos_id:
                break

        return sp.decode_ids(ys[0].tolist())


def load_image(img_path: str, img_size: int = 224) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img = Image.open(img_path).convert("RGB")
    return transform(img)


def main(args: argparse.Namespace):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    if not sp.load(args.tokenizer):
        raise FileNotFoundError(f"Tokenizer not found: {args.tokenizer}")

    # Build model
    encoder = ViTEncoder(model_name=args.vit_model, proj_dim=args.d_model, pretrained=False)
    decoder = TransformerDecoderModel(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_ff,
        max_len=args.max_len,
        pad_idx=3
    )
    model = Image2TextModel(encoder, decoder).to(device)

    # Load checkpoint
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load image
    img = load_image(args.image, args.img_size).to(device)

    # Generate (use beam search if beam_width > 1)
    if args.beam_width > 1:
        output_text = beam_search_decode(model, img, sp, beam_width=args.beam_width, max_len=args.max_len, device=device)
    else:
        output_text = greedy_decode(model, img, sp, max_len=args.max_len, device=device)
    print("✅ Translation:", output_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image-to-Text Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to tokenizer model")
    parser.add_argument("--vit_model", type=str, default="google/vit-base-patch16-224")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--dim_ff", type=int, default=2048)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--beam_width", type=int, default=4, help="Beam width for search (1 for greedy)")
    args = parser.parse_args()

    main(args)