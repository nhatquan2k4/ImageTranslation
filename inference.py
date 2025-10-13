import torch
import torch.nn as nn
import json
import os
from argparse import ArgumentParser
from modules.transformer import Transformer
from utils.tokenizer import tokenizer
from utils.translator import translate
from utils.image_processor import ImageProcessor


def main(args):
    model_folder = args.model_folder
    image_path = args.prompt

    print("Load config")
    config_file = open(os.path.join(model_folder, 'config.json'))
    cfg = json.load(config_file)
    max_strlen = cfg['max_strlen']
    k = cfg['k']
    image_size = cfg.get('image_size', 224)
    d_model = cfg.get('d_model', 512)
    heads = cfg.get('heads', 8)
    print(json.dumps(cfg, indent=3))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Use {device}")

    print("Creating tokenizer")
    trg_tokenizer = tokenizer('vi')

    print("Loading vocabulary")
    trg_vocab = torch.load(os.path.join(model_folder, 'trg_vocab.pth'))

    print("Creating image processor")
    image_processor = ImageProcessor(image_size=image_size)

    print("Loading model")
    model = Transformer(
        len(trg_vocab),
        d_model=d_model,
        n=cfg['n_layers'],
        heads=heads,
        dropout=cfg['dropout'],
        image_size=image_size,
        vision_hidden_dim=cfg.get('vision_hidden_dim', 768),
        vision_layers=cfg.get('vision_layers', 6),
        vision_heads=cfg.get('vision_heads', 8),
        patch_size=cfg.get('patch_size', 16)
    )

    model_file = os.path.join(model_folder, 'model_best.pt')
    model_ckpt = torch.load(model_file, map_location=torch.device(device))
    model.load_state_dict(model_ckpt)
    model = model.to(device)

    print("Running inference")
    print(f"Input image: {image_path}")
    print(f"Output: {translate(image_path, model, trg_vocab, max_strlen, device, k, image_processor)}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-p', '--prompt', type=str, required=True, help="Path to image file")
    parser.add_argument('-m', '--model_folder', type=str, required=True, help='Path to the model folder')
    args = parser.parse_args()
    main(args)
