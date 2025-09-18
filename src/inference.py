# inference.py
import argparse
import torch
from PIL import Image
import sentencepiece as spm
from torchvision import transforms
from model import ViTEncoder, TransformerDecoderModel, Image2TextModel


def greedy_decode(model, img, sp, max_len=128, device="cpu", pad_idx=3, bos_id=1, eos_id=2):
    model.eval()
    with torch.no_grad():
        # encoder forward
        memory = model.encoder(img.unsqueeze(0).to(device))  # (B=1, T, d_model)

        ys = torch.tensor([[bos_id]], dtype=torch.long, device=device)  # start with <bos>
        for i in range(max_len - 1):
            tgt_mask = (torch.triu(torch.ones(ys.size(1), ys.size(1), device=device)) == 1).transpose(0, 1)
            tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))

            out = model.decoder(ys, memory, tgt_mask=tgt_mask)
            out = out.transpose(0, 1)  # (1, T, vocab)
            prob = out[:, -1, :]  # last token
            next_word = torch.argmax(prob, dim=-1).item()

            ys = torch.cat([ys, torch.tensor([[next_word]], device=device)], dim=1)
            if next_word == eos_id:
                break

        return sp.decode_ids(ys[0].tolist())


def load_image(img_path, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
    img = Image.open(img_path).convert("RGB")
    return transform(img)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)

    # build model
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

    # load checkpoint
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # load image
    img = load_image(args.image, args.img_size).to(device)

    # generate
    output_text = greedy_decode(model, img, sp, max_len=args.max_len, device=device)
    print("✅ Translation:", output_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--vit_model", type=str, default="google/vit-base-patch16-224")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--dim_ff", type=int, default=2048)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--ckpt_path", type=str, required=True)
    args = parser.parse_args()

    main(args)
