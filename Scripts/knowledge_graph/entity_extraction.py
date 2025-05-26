"""
Entity extraction module for Knowledge Graph integration.
"""
from typing import List, Dict, Any, Tuple, Set, Optional
import re
import spacy
from collections import defaultdict

class EntityExtractor:
    """Extract entities from text for knowledge graph construction."""
    
    def __init__(self, model_name: str = "en_core_web_lg"):
        """Initialize the entity extractor.
        
        Args:
            model_name: Name of the spaCy model to use
        """
        # Load spaCy model
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            # If model not found, download it
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)
            self.nlp = spacy.load(model_name)
        
        # Entity types of interest
        self.entity_types = {
            "PERSON": "Person",
            "ORG": "Organization",
            "GPE": "Location",
            "LOC": "Location",
            "PRODUCT": "Product",
            "EVENT": "Event",
            "DATE": "Date",
            "TIME": "Time",
            "MONEY": "Money",
            "PERCENT": "Percentage",
            "WORK_OF_ART": "WorkOfArt",
            "LAW": "Law",
            "LANGUAGE": "Language"
        }
        
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text.
        
        Args:
            text: Input text to extract entities from
            
        Returns:
            List of extracted entities with their metadata
        """
        doc = self.nlp(text)
        
        entities = []
        for ent in doc.ents:
            # Skip entities with unwanted types
            if ent.label_ not in self.entity_types:
                continue
                
            entity = {
                "text": ent.text,
                "type": self.entity_types.get(ent.label_, ent.label_),
                "start": ent.start_char,
                "end": ent.end_char,
                "label": ent.label_
            }
            entities.append(entity)
            
        return entities
    
    def extract_relationships(self, text: str) -> List[Dict[str, Any]]:
        """Extract relationships between entities.
        
        Args:
            text: Input text to extract relationships from
            
        Returns:
            List of extracted relationships with their metadata
        """
        doc = self.nlp(text)
        
        # First, extract all entities
        entities = self.extract_entities(text)
        entity_spans = [(e["start"], e["end"], e) for e in entities]
        
        # Extract relationship triplets (subject, predicate, object)
        relationships = []
        
        # Process each sentence separately
        for sent in doc.sents:
            # Find the root verb
            root = None
            for token in sent:
                if token.dep_ == "ROOT" and token.pos_ == "VERB":
                    root = token
                    break
            
            if not root:
                continue
                
            # Find subject and object
            subject = None
            for child in root.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subject = child
                    break
            
            if not subject:
                continue
                
            # Find objects
            for child in root.children:
                if child.dep_ in ("dobj", "pobj", "attr"):
                    obj = child
                    
                    # Find entities that match the subject and object
                    subject_entity = self._find_entity_for_token(subject, entity_spans)
                    object_entity = self._find_entity_for_token(obj, entity_spans)
                    
                    if subject_entity and object_entity:
                        relationship = {
                            "subject": subject_entity,
                            "predicate": root.lemma_,
                            "object": object_entity,
                            "sentence": sent.text
                        }
                        relationships.append(relationship)
        
        return relationships
    
    def _find_entity_for_token(self, token, entity_spans):
        """Find an entity that contains the given token."""
        start = token.idx
        end = token.idx + len(token.text)
        
        # Find the closest entity that contains this token
        for e_start, e_end, entity in entity_spans:
            if start >= e_start and end <= e_end:
                return entity
            
        return None
