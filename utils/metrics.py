"""
Evaluation metrics for Image-to-Text models
"""
import random
import torch
from nltk.translate.bleu_score import corpus_bleu


def evaluate_bleu(model, val_dataset, tokenizer, config, num_samples=1000):
    """
    Evaluate BLEU score on validation dataset.
    
    Args:
        model: Trained model
        val_dataset: Validation dataset
        tokenizer: Tokenizer
        config: Configuration object
        num_samples: Number of samples to evaluate
    
    Returns:
        BLEU-4 score
    """
    from utils.beam_search import beam_search
    
    model.eval()
    val_size = len(val_dataset)
    num_eval = min(num_samples, val_size)
    eval_indices = random.sample(range(val_size), num_eval)
    
    references = []
    hypotheses = []
    
    print(f"\n{'-'*20} VALIDATION ({num_eval}/{val_size}) {'-'*20}")
    
    with torch.no_grad():
        for i, idx in enumerate(eval_indices):
            image, _, target_text, _ = val_dataset[idx]
            
            # Generate prediction
            with torch.cuda.amp.autocast(enabled=config.use_amp):
                image = image.unsqueeze(0).to(config.device if hasattr(config, 'device') else 'cuda')
                pred_tokens = beam_search(
                    model, image, tokenizer,
                    beam_size=config.beam_size,
                    max_len=config.max_length,
                    length_penalty=config.length_penalty,
                    repetition_penalty=config.repetition_penalty
                )
            
            # Decode prediction
            pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
            
            # Store for BLEU computation
            references.append([target_text.lower().split()])
            hypotheses.append(pred_text.lower().split())
            
            # Print samples
            if i < 2:
                print(f"[Sample {i+1}]")
                print(f"  Target: {target_text}")
                print(f"  Pred  : {pred_text}\n")
            
            if (i + 1) % 500 == 0:
                print(f"  ... {i+1}/{num_eval} done")
    
    # Compute BLEU
    bleu4 = corpus_bleu(references, hypotheses)
    print(f">>> ✅ BLEU-4: {bleu4:.4f}")
    print(f"{'-'*60}\n")
    
    model.train()
    return bleu4


def evaluate_with_examples(model, val_dataset, tokenizer, config, num_examples=5):
    """
    Evaluate and show example predictions.
    
    Args:
        model: Trained model
        val_dataset: Validation dataset
        tokenizer: Tokenizer
        config: Configuration object
        num_examples: Number of examples to show
    """
    from utils.beam_search import beam_search
    
    model.eval()
    sample_indices = random.sample(range(len(val_dataset)), num_examples)
    
    print("\n🔍 SAMPLE PREDICTIONS:")
    
    for i, idx in enumerate(sample_indices):
        image, _, target_text, _ = val_dataset[idx]
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=config.use_amp):
                image = image.unsqueeze(0).to(config.device if hasattr(config, 'device') else 'cuda')
                pred_tokens = beam_search(
                    model, image, tokenizer,
                    beam_size=config.beam_size,
                    max_len=config.max_length,
                    length_penalty=config.length_penalty,
                    repetition_penalty=config.repetition_penalty
                )
        
        pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
        
        print(f"\n[Example {i+1}]")
        print(f"Target: {target_text}")
        print(f"Prediction: {pred_text}")
    
    print()


def compute_accuracy(predictions, targets):
    """
    Compute word-level accuracy.
    
    Args:
        predictions: List of predicted texts
        targets: List of target texts
    
    Returns:
        Accuracy score
    """
    correct = 0
    total = 0
    
    for pred, target in zip(predictions, targets):
        pred_words = pred.lower().split()
        target_words = target.lower().split()
        
        for pw, tw in zip(pred_words, target_words):
            if pw == tw:
                correct += 1
            total += 1
    
    return correct / total if total > 0 else 0.0
