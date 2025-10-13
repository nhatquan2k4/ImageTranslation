import re

# Import Vietnamese tokenizer
try:
    from underthesea import word_tokenize
    HAS_UNDERTHESEA = True
except ImportError:
    HAS_UNDERTHESEA = False

# Import English tokenizer
try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

class tokenizer(object):
    def __init__(self, lang):
        """
        Tokenizer for image-to-translation task
        
        Args:
            lang: 'vi' for Vietnamese (target), 'en' for English (source, if needed)
        """
        self.lang = lang
        
        if lang == "en":
            if not HAS_SPACY:
                raise ImportError("Need spacy for English: pip install spacy && python -m spacy download en_core_web_sm")
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                raise ImportError("Please install English model: python -m spacy download en_core_web_sm")
                
        elif lang == "vi":
            if not HAS_UNDERTHESEA:
                raise ImportError("Need underthesea for Vietnamese: pip install underthesea")
            self.nlp = None  # Use underthesea instead of spacy
            
        else:
            raise ValueError("Only 'vi' and 'en' are supported")

    def clean_text(self, sentence):
        """
        Clean and normalize text
        """
        # Remove special characters but keep Vietnamese diacritics
        sentence = re.sub(r'[\*\"""\n\\…\+\-\/\=\(\)\'•:\[\]\|\'!;]', " ", sentence)
        # Normalize multiple spaces
        sentence = re.sub(r"[ ]+", " ", sentence)
        # Normalize punctuation
        sentence = re.sub(r"\!+", "!", sentence)
        sentence = re.sub(r"\,+", ",", sentence)
        sentence = re.sub(r"\?+", "?", sentence)
        sentence = re.sub(r"\.+", ".", sentence)
        
        # For Vietnamese, preserve case to maintain proper nouns
        if self.lang == "vi":
            return sentence.strip()
        else:
            return sentence.lower().strip()

    def tokenize(self, sentence):
        """
        Tokenize sentence into list of tokens
        
        Args:
            sentence: Input text string
            
        Returns:
            list: List of tokens
        """
        if not sentence or not isinstance(sentence, str):
            return []
            
        sentence = self.clean_text(sentence)
        if not sentence:
            return []
            
        tokens = []

        if self.lang == "vi":
            # Use underthesea for Vietnamese
            try:
                tokens = word_tokenize(sentence, format="text").split()
            except Exception as e:
                print(f"Error tokenizing Vietnamese text: {e}")
                # Fallback to simple split
                tokens = sentence.split()
                
        elif self.lang == "en":
            # Use spacy for English
            try:
                for tok in self.nlp.tokenizer(sentence):
                    if tok.text.strip():  # Skip empty tokens
                        tokens.append(tok.text)
            except Exception as e:
                print(f"Error tokenizing English text: {e}")
                # Fallback to simple split
                tokens = sentence.split()

        return tokens
