import torch
import re
from utils.beam_search import beam_search
from utils.image_processor import ImageProcessor
from PIL import Image


def multiple_replace(mapping, text):
    """
    Apply multiple string replacements
    """
    regex = re.compile("(%s)" % "|".join(map(re.escape, mapping.keys())))
    return regex.sub(lambda mo: mapping[mo.string[mo.start():mo.end()]], text)


def convert_to_word(idx_list, vocab):
    """
    Convert list of token indices to text
    """
    # Tạo inverse vocabulary mapping
    if hasattr(vocab, "get_itos"):
        # torchtext.vocab.vocab object
        itos = vocab.get_itos()
        word_list = [itos[tok] for tok in idx_list if tok < len(itos)]
    elif hasattr(vocab, "itos"):
        # Legacy torchtext vocab object
        word_list = [vocab.itos[tok] for tok in idx_list if tok in vocab.itos]
    else:
        # Dictionary vocab {token: idx}
        inv_vocab = {v: k for k, v in vocab.items()}
        word_list = [inv_vocab[tok] for tok in idx_list if tok in inv_vocab]
    
    # Filter out special tokens
    filtered_words = []
    for word in word_list:
        if word not in ['<sos>', '<eos>', '<pad>', '<unk>']:
            filtered_words.append(word)
        elif word == '<eos>':
            break  # Stop at end of sentence
    
    return ' '.join(filtered_words)


def translate(image_path, model, vocab, max_strlen, device, k, image_processor):
    """
    Translate text trong ảnh sang tiếng Việt
    
    Args:
        image_path: Path to input image containing English text
        model: Trained image-to-translation model
        vocab: Target vocabulary (Vietnamese)
        max_strlen: Maximum output sequence length
        device: torch.device
        k: Beam search width
        image_processor: ImageProcessor instance
    
    Returns:
        str: Vietnamese translation of text in image
    """
    model.eval()

    try:
        # Load và process image
        if isinstance(image_path, str):
            # Load from file path
            img_tensor = image_processor.process_for_inference(image_path)
        else:
            # Assume it's already a PIL Image
            img_tensor = image_processor.process_image(image_path, is_training=False)
        
        if img_tensor is None:
            return "Lỗi xử lý ảnh"

        # Add batch dimension và move to device
        img_tensor = img_tensor.unsqueeze(0).to(device)

        # Beam search translation
        with torch.no_grad():
            sentence, length = beam_search(img_tensor, model, vocab, device, k, max_strlen)
        
        # Convert indices to words
        if length > 1:
            sentence = sentence[1:length]  # Remove <sos> token
        else:
            sentence = sentence[1:]  # fallback
            
        translation = convert_to_word(sentence, vocab)
        
        # Post-processing để clean up punctuation
        translation = multiple_replace({
            ' ?': '?',
            ' !': '!',
            ' .': '.',
            '\' ': '\'',
            ' ,': ',',
            ' ;': ';',
            ' :': ':',
            '  ': ' '  # Remove double spaces
        }, translation)
        
        return translation.strip()
        
    except Exception as e:
        print(f"Error during translation: {e}")
        return f"Lỗi dịch: {str(e)}"
