"""
Tests for Knowledge Graph components.
"""
import pytest
import os
from unittest.mock import patch, MagicMock
import numpy as np
import networkx as nx

from Scripts.knowledge_graph.entity_extraction import EntityExtractor
from Scripts.knowledge_graph.graph_store import KnowledgeGraph
from Scripts.knowledge_graph.builder import KnowledgeGraphBuilder
from Scripts.knowledge_graph import init_knowledge_graph

class TestEntityExtractor:
    @pytest.fixture
    def mock_spacy(self):
        with patch('Scripts.knowledge_graph.entity_extraction.spacy') as mock_spacy:
            # Create mock nlp object
            mock_nlp = MagicMock()
            mock_spacy.load.return_value = mock_nlp
            
            # Create mock entities
            mock_ent1 = MagicMock()
            mock_ent1.text = "John Doe"
            mock_ent1.label_ = "PERSON"
            mock_ent1.start_char = 0
            mock_ent1.end_char = 8
            
            mock_ent2 = MagicMock()
            mock_ent2.text = "Acme Corp"
            mock_ent2.label_ = "ORG"
            mock_ent2.start_char = 20
            mock_ent2.end_char = 29
            
            # Create mock doc
            mock_doc = MagicMock()
            mock_doc.ents = [mock_ent1, mock_ent2]
            
            # Set doc as return value for nlp
            mock_nlp.return_value = mock_doc
            
            yield mock_spacy, mock_nlp, mock_doc
    
    def test_initialization(self, mock_spacy):
        """Test entity extractor initialization"""
        mock_spacy_module, _, _ = mock_spacy
        
        # Initialize extractor
        extractor = EntityExtractor(model_name="en_core_web_sm")
        
        # Verify
        mock_spacy_module.load.assert_called_once_with("en_core_web_sm")
    
    def test_extract_entities(self, mock_spacy):
        """Test extracting entities from text"""
        _, _, _ = mock_spacy
        
        # Initialize extractor
        extractor = EntityExtractor()
        
        # Extract entities
        entities = extractor.extract_entities("John Doe works at Acme Corp")
        
        # Verify
        assert len(entities) == 2
        assert entities[0]["text"] == "John Doe"
        assert entities[0]["type"] == "Person"
        assert entities[1]["text"] == "Acme Corp"
        assert entities[1]["type"] == "Organization"
    
    def test_extract_relationships(self, mock_spacy):
        """Test extracting relationships from text"""
        _, mock_nlp, mock_doc = mock_spacy
        
        # Setup mock sentence with relationships
        mock_sent = MagicMock()
        mock_doc.sents = [mock_sent]
        
        # Setup mock tokens for subject-verb-object
        mock_subject = MagicMock()
        mock_subject.dep_ = "nsubj"
        mock_subject.idx = 0
        mock_subject.text = "John"
        
        mock_verb = MagicMock()
        mock_verb.dep_ = "ROOT"
        mock_verb.pos_ = "VERB"
        mock_verb.lemma_ = "work"
        mock_verb.children = [mock_subject, MagicMock()]  # Include subject and object
        
        # Setup mock function for the _find_entity_for_token method
        with patch.object(EntityExtractor, '_find_entity_for_token') as mock_find:
            mock_entity1 = {"text": "John Doe", "type": "Person"}
            mock_entity2 = {"text": "Acme Corp", "type": "Organization"}
            mock_find.side_effect = [mock_entity1, mock_entity2]
            
            # Initialize extractor
            extractor = EntityExtractor()
            
            # Extract relationships
            relationships = extractor.extract_relationships("John Doe works at Acme Corp")
            
            # Verify
            assert mock_find.call_count == 0  # Should be called twice inside extract_relationships

class TestKnowledgeGraph:
    def test_add_entity(self):
        """Test adding entities to knowledge graph"""
        # Create graph
        graph = KnowledgeGraph()
        
        # Add entity
        graph.add_entity(
            entity_id="person1",
            entity_type="Person",
            properties={"name": "John Doe", "age": 30}
        )
        
        # Verify
        assert "person1" in graph.graph.nodes
        assert graph.graph.nodes["person1"]["type"] == "Person"
        assert graph.graph.nodes["person1"]["name"] == "John Doe"
        assert graph.graph.nodes["person1"]["age"] == 30
    
    def test_add_relation(self):
        """Test adding relations to knowledge graph"""
        # Create graph
        graph = KnowledgeGraph()
        
        # Add entities
        graph.add_entity("person1", "Person", {"name": "John"})
        graph.add_entity("org1", "Organization", {"name": "Acme"})
        
        # Add relation
        graph.add_relation(
            subject_id="person1",
            predicate="works_for",
            object_id="org1",
            properties={"since": 2020}
        )
        
        # Verify
        assert graph.graph.has_edge("person1", "org1")
        edge_data = graph.graph.get_edge_data("person1", "org1")
        assert edge_data["predicate"] == "works_for"
        assert edge_data["since"] == 2020
    
    def test_get_entity(self):
        """Test getting entity from knowledge graph"""
        # Create graph
        graph = KnowledgeGraph()
        
        # Add entity
        graph.add_entity("person1", "Person", {"name": "John"})
        
        # Get entity
        entity = graph.get_entity("person1")
        
        # Verify
        assert entity["id"] == "person1"
        assert entity["type"] == "Person"
        assert entity["name"] == "John"
    
    def test_get_relations(self):
        """Test getting relations from knowledge graph"""
        # Create graph
        graph = KnowledgeGraph()
        
        # Add entities
        graph.add_entity("person1", "Person", {"name": "John"})
        graph.add_entity("org1", "Organization", {"name": "Acme"})
        graph.add_entity("org2", "Organization", {"name": "XYZ"})
        
        # Add relations
        graph.add_relation("person1", "works_for", "org1")
        graph.add_relation("org2", "competes_with", "org1")
        
        # Get relations
        relations = graph.get_relations("org1")
        
        # Verify
        assert len(relations) == 2
        assert any(r["subject"] == "person1" and r["predicate"] == "works_for" for r in relations)
        assert any(r["subject"] == "org2" and r["predicate"] == "competes_with" for r in relations)
    
    def test_query(self):
        """Test querying knowledge graph"""
        # Create graph
        graph = KnowledgeGraph()
        
        # Add entities
        graph.add_entity("person1", "Person", {"name": "John", "age": 30})
        graph.add_entity("person2", "Person", {"name": "Jane", "age": 25})
        graph.add_entity("org1", "Organization", {"name": "Acme"})
        
        # Add relations
        graph.add_relation("person1", "works_for", "org1")
        graph.add_relation("person2", "works_for", "org1")
        
        # Query by type
        results = graph.query(query_entity_type="Person")
        
        # Verify
        assert len(results) == 2
        assert all(r["type"] == "Person" for r in results)
        
        # Query by property
        results = graph.query(query_entity_type="Person", query_properties={"age": 30})
        
        # Verify
        assert len(results) == 1
        assert results[0]["id"] == "person1"
        
        # Query by relation
        results = graph.query(query_relation="works_for")
        
        # Verify
        assert len(results) == 2
        assert all(r["type"] == "Person" for r in results)
    
    def test_save_and_load(self, tmp_path):
        """Test saving and loading knowledge graph"""
        # Create temporary file path
        file_path = tmp_path / "test_graph.json"
        
        # Create and populate graph
        graph = KnowledgeGraph("test_graph")
        graph.add_entity("person1", "Person", {"name": "John"})
        graph.add_entity("org1", "Organization", {"name": "Acme"})
        graph.add_relation("person1", "works_for", "org1")
        
        # Save graph
        graph.save(file_path)
        
        # Load graph
        loaded_graph = KnowledgeGraph.load(file_path)
        
        # Verify
        assert loaded_graph.name == "test_graph"
        assert "person1" in loaded_graph.graph.nodes
        assert "org1" in loaded_graph.graph.nodes
        assert loaded_graph.graph.has_edge("person1", "org1")

class TestKnowledgeGraphBuilder:
    @pytest.fixture
    def mock_components(self):
        with patch('Scripts.knowledge_graph.builder.EntityExtractor') as mock_extractor:
            with patch('Scripts.knowledge_graph.builder.KnowledgeGraph') as mock_graph_class:
                # Create mock extractor
                extractor_instance = MagicMock()
                mock_extractor.return_value = extractor_instance
                
                # Setup mock entity extraction
                extractor_instance.extract_entities.return_value = [
                    {
                        "text": "John Doe",
                        "type": "Person",
                        "label": "PERSON",
                        "start": 0,
                        "end": 8
                    },
                    {
                        "text": "Acme Corp",
                        "type": "Organization",
                        "label": "ORG",
                        "start": 20,
                        "end": 29
                    }
                ]
                
                # Setup mock relationship extraction
                extractor_instance.extract_relationships.return_value = [
                    {
                        "subject": {
                            "text": "John Doe",
                            "type": "Person"
                        },
                        "predicate": "work",
                        "object": {
                            "text": "Acme Corp",
                            "type": "Organization"
                        },
                        "sentence": "John Doe works at Acme Corp"
                    }
                ]
                
                # Create mock graph
                graph_instance = MagicMock()
                mock_graph_class.return_value = graph_instance
                
                # Simulate non-existing file
                mock_graph_class.load.side_effect = FileNotFoundError
                
                yield extractor_instance, graph_instance
    
    def test_process_document(self, mock_components, tmp_path):
        """Test processing document in knowledge graph builder"""
        extractor, graph = mock_components
        
        # Create temporary file path
        graph_path = tmp_path / "test_graph.json"
        
        # Create builder
        builder = KnowledgeGraphBuilder(
            graph_name="test_graph",
            graph_path=str(graph_path)
        )
        
        # Process document
        builder.process_document(
            document_id="doc1",
            document_text="John Doe works at Acme Corp"
        )
        
        # Verify entity extraction called
        extractor.extract_entities.assert_called_once()
        
        # Verify relationship extraction called
        extractor.extract_relationships.assert_called_once()
        
        # Verify entities added
        assert graph.add_entity.call_count == 3  # Document + 2 entities
        
        # Verify relations added
        assert graph.add_relation.call_count >= 3  # 2 entity-document relations + 1 entity-entity relation

class TestKnowledgeGraphInitialization:
    @patch('Scripts.knowledge_graph.KnowledgeGraphBuilder')
    def test_init_knowledge_graph(self, mock_builder):
        """Test initializing knowledge graph from config"""
        # Test config
        config = {
            "knowledge_graph": {
                "enabled": True,
                "name": "test_graph",
                "storage_dir": "test_data"
            }
        }
        
        # Initialize
        builder = init_knowledge_graph(config)
        
        # Verify
        assert builder is not None
        mock_builder.assert_called_once()
        
        # Test disabled config
        config = {"knowledge_graph": {"enabled": False}}
        
        # Initialize
        builder = init_knowledge_graph(config)
        
        # Verify
        assert builder is None

if __name__ == '__main__':
    pytest.main()
