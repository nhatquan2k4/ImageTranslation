import torch
import torch.nn as nn
from collections import defaultdict
import editdistance
import re

class ValidationMetrics:
    def __init__(self, vocab, tokenizer, device):
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.device = device
        self.idx_to_token = {v: k for k, v in vocab.items()}
        
        # Don't initialize translator to avoid circular import
        # self.translator = Translator()
    
    def calculate_metrics(self, model, dataloader, max_samples=None):
        """
        Calculate comprehensive validation metrics
        """
        model.eval()
        metrics = defaultdict(list)
        
        total_samples = 0
        correct_predictions = 0
        total_loss = 0
        
        # For BLEU calculation
        references = []
        hypotheses = []
        
        # For CER/WER calculation
        cer_scores = []
        wer_scores = []
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(dataloader):
                if max_samples and total_samples >= max_samples:
                    break
                
                batch_size = images.size(0)
                
                # Generate predictions using simple greedy decoding
                predictions = self._simple_decode(model, images, max_length=targets.size(1))
                
                # Convert targets to text
                target_texts = []
                for i in range(batch_size):
                    target_tokens = []
                    for idx in targets[i]:
                        if idx.item() == self.vocab['<pad>']:
                            break
                        if idx.item() not in [self.vocab['<sos>'], self.vocab['<eos>']]:
                            target_tokens.append(self.idx_to_token[idx.item()])
                    target_texts.append(' '.join(target_tokens))
                
                # Calculate metrics for each sample in batch
                for i in range(batch_size):
                    pred_text = predictions[i] if i < len(predictions) else ""
                    target_text = target_texts[i]
                    
                    # Exact match accuracy
                    if self._normalize_text(pred_text) == self._normalize_text(target_text):
                        correct_predictions += 1
                    
                    # Store for BLEU
                    references.append([target_text.split()])
                    hypotheses.append(pred_text.split())
                    
                    # CER (Character Error Rate)
                    cer = editdistance.eval(pred_text, target_text) / max(len(target_text), 1)
                    cer_scores.append(cer)
                    
                    # WER (Word Error Rate)
                    pred_words = pred_text.split()
                    target_words = target_text.split()
                    wer = editdistance.eval(pred_words, target_words) / max(len(target_words), 1)
                    wer_scores.append(wer)
                
                total_samples += batch_size
                
                # Progress
                if batch_idx % 10 == 0:
                    print(f'Validation batch {batch_idx}/{len(dataloader)}')
        
        # Calculate final metrics
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        avg_cer = sum(cer_scores) / len(cer_scores) if cer_scores else 0
        avg_wer = sum(wer_scores) / len(wer_scores) if wer_scores else 0
        
        # BLEU score calculation
        bleu_score = self._calculate_bleu(references, hypotheses)
        
        return {
            'accuracy': accuracy,
            'cer': avg_cer,
            'wer': avg_wer,
            'bleu': bleu_score,
            'total_samples': total_samples
        }
    
    def _normalize_text(self, text):
        """Normalize text for comparison"""
        return re.sub(r'\s+', ' ', text.strip().lower())
    
    def _calculate_bleu(self, references, hypotheses):
        """
        Simple BLEU score calculation
        """
        try:
            from nltk.translate.bleu_score import corpus_bleu
            return corpus_bleu(references, hypotheses)
        except ImportError:
            # Fallback simple n-gram precision
            return self._simple_bleu(references, hypotheses)
    
    def _simple_bleu(self, references, hypotheses):
        """
        Simplified BLEU calculation without NLTK
        """
        if not references or not hypotheses:
            return 0.0
        
        # 1-gram precision
        total_matches = 0
        total_pred_words = 0
        
        for ref, hyp in zip(references, hypotheses):
            ref_words = set(ref[0]) if ref and ref[0] else set()
            hyp_words = hyp if hyp else []
            
            matches = sum(1 for word in hyp_words if word in ref_words)
            total_matches += matches
            total_pred_words += len(hyp_words)
        
        precision = total_matches / total_pred_words if total_pred_words > 0 else 0
        return precision


class EarlyStopping:
    """
    Early stopping utility
    """
    def __init__(self, patience=5, min_delta=0.001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        
    def __call__(self, score):
        improved = False
        
        if self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
                improved = True
            else:
                self.counter += 1
        else:
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
                improved = True
            else:
                self.counter += 1
        
        return {
            'improved': improved,
            'should_stop': self.counter >= self.patience,
            'counter': self.counter,
            'best_score': self.best_score
        }
    
    def _simple_decode(self, model, images, max_length=50):
        """
        Simple greedy decoding without beam search to avoid circular imports
        """
        model.eval()
        batch_size = images.size(0)
        predictions = []
        
        # Get vocabulary indices - handle both dict and vocab object
        if isinstance(self.vocab, dict):
            sos_idx = self.vocab.get('<sos>', 1)
            eos_idx = self.vocab.get('<eos>', 2)
            pad_idx = self.vocab.get('<pad>', 0)
        else:
            sos_idx = self.vocab['<sos>'] if '<sos>' in self.vocab else 1
            eos_idx = self.vocab['<eos>'] if '<eos>' in self.vocab else 2
            pad_idx = self.vocab['<pad>'] if '<pad>' in self.vocab else 0
        
        with torch.no_grad():
            for i in range(batch_size):
                # Single image
                img = images[i:i+1]
                
                # Start with SOS token
                generated = [sos_idx]
                
                for _ in range(max_length):
                    # Create input sequence
                    input_seq = torch.tensor([generated], device=self.device)
                    
                    # Create target mask (for decoder)
                    tgt_len = len(generated)
                    tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=self.device))
                    tgt_mask = tgt_mask.unsqueeze(0)
                    
                    try:
                        # Forward pass
                        output = model(img, input_seq, tgt_mask)
                        
                        # Get next token (greedy)
                        next_token_logits = output[0, -1, :]  # Last token prediction
                        next_token = torch.argmax(next_token_logits).item()
                        
                        # Stop if EOS token
                        if next_token == eos_idx:
                            break
                            
                        generated.append(next_token)
                        
                    except Exception as e:
                        print(f"Error in decoding: {e}")
                        break
                
                # Convert to text
                pred_text = self._indices_to_text(generated)
                predictions.append(pred_text)
        
        return predictions
    
    def _indices_to_text(self, indices):
        """Convert token indices to text"""
        tokens = []
        for idx in indices:
            if idx in self.idx_to_token:
                token = self.idx_to_token[idx]
                if token not in ['<sos>', '<eos>', '<pad>', '<unk>']:
                    tokens.append(token)
        
        return ' '.join(tokens)