from typing import List, Dict, Any, Optional
import re
import unicodedata
import ftfy
from bs4 import BeautifulSoup
import spacy
from langdetect import detect
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TextCleaningConfig:
    remove_urls: bool = True
    remove_emails: bool = True
    fix_unicode: bool = True
    normalize_whitespace: bool = True
    fix_punctuation: bool = True
    remove_brackets: bool = False
    fix_accents: bool = True
    fix_line_breaks: bool = True
    remove_special_characters: bool = True
    normalize_numbers: bool = True
    remove_stopwords: bool = False
    lemmatize: bool = False

class AdvancedTextCleaner:
    def __init__(self, config: TextCleaningConfig = None):
        self.config = config or TextCleaningConfig()
        if self.config.lemmatize:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("Spacy model not found. Installing en_core_web_sm...")
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
                self.nlp = spacy.load("en_core_web_sm")

    def clean_text(self, text: str) -> str:
        if not text:
            return ""

        # Detect language
        try:
            lang = detect(text)
            logger.debug(f"Detected language: {lang}")
        except:
            lang = "en"
            logger.warning("Language detection failed, defaulting to English")

        # Fix Unicode issues first
        if self.config.fix_unicode:
            text = ftfy.fix_text(text)

        # Remove HTML tags if any
        text = BeautifulSoup(text, "html.parser").get_text()

        # URL removal
        if self.config.remove_urls:
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

        # Email removal
        if self.config.remove_emails:
            text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '', text)

        # Normalize whitespace
        if self.config.normalize_whitespace:
            text = ' '.join(text.split())

        # Fix punctuation
        if self.config.fix_punctuation:
            text = re.sub(r'([.,!?()])\1+', r'\1', text)  # Remove repeated punctuation
            text = re.sub(r'\s([.,!?()])', r'\1', text)   # Remove space before punctuation

        # Remove brackets and their contents
        if self.config.remove_brackets:
            text = re.sub(r'\[.*?\]|\(.*?\)|\{.*?\}', '', text)

        # Fix accents
        if self.config.fix_accents:
            text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')

        # Fix line breaks
        if self.config.fix_line_breaks:
            text = re.sub(r'\n+', ' ', text)

        # Remove special characters
        if self.config.remove_special_characters:
            text = re.sub(r'[^\w\s.,!?()-]', '', text)

        # Normalize numbers
        if self.config.normalize_numbers:
            text = re.sub(r'\d+\.\d+', 'DECIMAL_NUMBER', text)
            text = re.sub(r'\d+', 'NUMBER', text)

        # Lemmatization and stopword removal
        if self.config.lemmatize:
            doc = self.nlp(text)
            if self.config.remove_stopwords:
                text = ' '.join([token.lemma_ for token in doc if not token.is_stop])
            else:
                text = ' '.join([token.lemma_ for token in doc])

        return text.strip()

    def clean_batch(self, texts: List[str], batch_size: int = 1000) -> List[str]:
        """Process a batch of texts in parallel"""
        from concurrent.futures import ThreadPoolExecutor
        
        cleaned_texts = []
        with ThreadPoolExecutor() as executor:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                cleaned_batch = list(executor.map(self.clean_text, batch))
                cleaned_texts.extend(cleaned_batch)
        
        return cleaned_texts

    @staticmethod
    def detect_language(text: str) -> str:
        try:
            return detect(text)
        except:
            return "unknown"
