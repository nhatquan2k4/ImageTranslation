# train.py
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sentencepiece as spm
from model import ViTEncoder, TransformerDecoderModel, Image2TextModel
from dataset import ImgToTextDataset, collate_batch
from utils import save_checkpoint
from tqdm import tqdm


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load tokenizer (sentencepiece)
    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)

    # build dataset
    dataset = ImgToTextDataset(
        args.data_json,
        tokenizer=sp,
        max_len=args.max_len,
        img_size=args.img_size
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=4
    )

    # model
    encoder = ViTEncoder(
        model_name=args.vit_model,
        proj_dim=args.d_model,
        pretrained=True
    )
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=3)

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(dataloader)
        total_loss = 0.0

        for imgs, tgts in pbar:
            imgs = imgs.to(device)
            tgts = tgts.to(device)

            decoder_input = tgts[:, :-1]
            labels = tgts[:, 1:]

            T = decoder_input.size(1)
            tgt_mask = generate_square_subsequent_mask(T).to(device)
            tgt_key_padding_mask = (decoder_input == 3)

            logits = model(
                imgs,
                decoder_input,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )
            # logits: (T, B, V)
            logits = logits.transpose(0, 1).contiguous().view(-1, logits.size(-1))
            labels_flat = labels.contiguous().view(-1)

            loss = criterion(logits, labels_flat)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_description(f"Epoch {epoch} Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")

        # save checkpoint
        save_checkpoint(model, optimizer, epoch, avg_loss, args.ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_json", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--vit_model", type=str, default="google/vit-base-patch16-224")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--dim_ff", type=int, default=2048)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/model.pt")
    args = parser.parse_args()

    train(args)
