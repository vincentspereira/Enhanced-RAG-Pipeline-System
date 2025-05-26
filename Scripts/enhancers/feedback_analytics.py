"""
Feedback analytics and automated model fine-tuning based on user feedback.

This module implements collection, analysis, and visualization of user feedback 
on search results and supports automated fine-tuning of models based on feedback.
"""
from typing import List, Dict, Any, Optional, Union, Set, Tuple
from dataclasses import dataclass, field
import logging
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import pickle
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import torch
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

@dataclass
class FeedbackEntry:
    """Represents a single feedback entry"""
    query: str
    document_id: str
    document_text: str
    relevance_score: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedbackAnalyticsConfig:
    """Configuration for feedback analytics"""
    storage_path: str = "data/feedback"
    min_feedback_threshold: int = 10
    auto_finetune_threshold: int = 100
    enable_auto_finetune: bool = True
    finetune_schedule: str = "weekly"  # daily, weekly, monthly
    finetune_models_dir: str = "data/models/finetuned"
    evaluation_split: float = 0.2
    training_epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    max_training_samples: int = 10000
    min_feedback_confidence: float = 0.6
    feedback_retention_days: int = 365
    aggregation_window: str = "day"  # hour, day, week, month


class FeedbackAnalytics:
    """Analytics for search result feedback"""
    
    def __init__(self, config: FeedbackAnalyticsConfig = None):
        """Initialize feedback analytics.
        
        Args:
            config: Configuration for feedback analytics
        """
        self.config = config or FeedbackAnalyticsConfig()
        self.feedback_data: List[FeedbackEntry] = []
        self._load_feedback_data()
    
    def _load_feedback_data(self):
        """Load feedback data from disk"""
        storage_path = Path(self.config.storage_path)
        if not storage_path.exists():
            storage_path.mkdir(parents=True, exist_ok=True)
            return
        
        feedback_file = storage_path / "feedback_data.pkl"
        if not feedback_file.exists():
            return
        
        try:
            with open(feedback_file, "rb") as f:
                self.feedback_data = pickle.load(f)
            
            # Filter out old feedback if retention period is set
            if self.config.feedback_retention_days > 0:
                cutoff_date = datetime.now() - timedelta(days=self.config.feedback_retention_days)
                self.feedback_data = [
                    entry for entry in self.feedback_data
                    if datetime.fromisoformat(entry.timestamp) >= cutoff_date
                ]
            
            logger.info(f"Loaded {len(self.feedback_data)} feedback entries")
        except Exception as e:
            logger.error(f"Failed to load feedback data: {e}")
    
    def _save_feedback_data(self):
        """Save feedback data to disk"""
        storage_path = Path(self.config.storage_path)
        storage_path.mkdir(parents=True, exist_ok=True)
        
        feedback_file = storage_path / "feedback_data.pkl"
        try:
            with open(feedback_file, "wb") as f:
                pickle.dump(self.feedback_data, f)
            logger.info(f"Saved {len(self.feedback_data)} feedback entries")
        except Exception as e:
            logger.error(f"Failed to save feedback data: {e}")
    
    def add_feedback(self, feedback: FeedbackEntry):
        """Add a new feedback entry.
        
        Args:
            feedback: Feedback entry to add
        """
        self.feedback_data.append(feedback)
        self._save_feedback_data()
        
        # Check if auto-finetune should be triggered
        if (self.config.enable_auto_finetune and 
            len(self.feedback_data) % self.config.auto_finetune_threshold == 0):
            self.trigger_model_finetune()
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get feedback statistics.
        
        Returns:
            Dictionary of feedback statistics
        """
        if not self.feedback_data:
            return {
                "total_feedback": 0,
                "average_relevance": 0,
                "positive_feedback": 0,
                "negative_feedback": 0
            }
        
        total = len(self.feedback_data)
        avg_relevance = sum(entry.relevance_score for entry in self.feedback_data) / total
        positive = sum(1 for entry in self.feedback_data if entry.relevance_score >= 0.5)
        negative = total - positive
        
        # Get timeframe statistics
        current_time = datetime.now()
        last_day = current_time - timedelta(days=1)
        last_week = current_time - timedelta(days=7)
        last_month = current_time - timedelta(days=30)
        
        day_count = sum(1 for entry in self.feedback_data 
                      if datetime.fromisoformat(entry.timestamp) >= last_day)
        week_count = sum(1 for entry in self.feedback_data 
                       if datetime.fromisoformat(entry.timestamp) >= last_week)
        month_count = sum(1 for entry in self.feedback_data 
                        if datetime.fromisoformat(entry.timestamp) >= last_month)
        
        return {
            "total_feedback": total,
            "average_relevance": avg_relevance,
            "positive_feedback": positive,
            "negative_feedback": negative,
            "last_day_count": day_count,
            "last_week_count": week_count,
            "last_month_count": month_count
        }
    
    def get_feedback_timeline(self) -> Dict[str, Any]:
        """Get feedback timeline data.
        
        Returns:
            Dictionary with timeline data
        """
        if not self.feedback_data:
            return {"timeline": []}
        
        # Determine aggregation format
        if self.config.aggregation_window == "hour":
            date_format = "%Y-%m-%d %H:00"
            timedelta_args = {"hours": 1}
        elif self.config.aggregation_window == "week":
            date_format = "%Y-%W"
            timedelta_args = {"days": 7}
        elif self.config.aggregation_window == "month":
            date_format = "%Y-%m"
            timedelta_args = {"days": 30}
        else:  # default to day
            date_format = "%Y-%m-%d"
            timedelta_args = {"days": 1}
        
        # Aggregate feedback by time period
        timeline_data = defaultdict(lambda: {"count": 0, "avg_score": 0, "positive": 0, "negative": 0})
        
        for entry in self.feedback_data:
            entry_time = datetime.fromisoformat(entry.timestamp)
            period = entry_time.strftime(date_format)
            
            timeline_data[period]["count"] += 1
            timeline_data[period]["avg_score"] = (
                (timeline_data[period]["avg_score"] * (timeline_data[period]["count"] - 1) + 
                 entry.relevance_score) / timeline_data[period]["count"]
            )
            
            if entry.relevance_score >= 0.5:
                timeline_data[period]["positive"] += 1
            else:
                timeline_data[period]["negative"] += 1
        
        # Convert to sorted list
        sorted_data = [
            {
                "period": period,
                "count": data["count"],
                "avg_score": data["avg_score"],
                "positive": data["positive"],
                "negative": data["negative"]
            }
            for period, data in sorted(timeline_data.items())
        ]
        
        return {"timeline": sorted_data}
    
    def get_query_analysis(self) -> Dict[str, Any]:
        """Analyze queries with the most feedback.
        
        Returns:
            Dictionary with query analysis
        """
        if not self.feedback_data:
            return {"queries": []}
        
        # Group feedback by query
        query_data = defaultdict(lambda: {"count": 0, "avg_score": 0, "positive": 0, "negative": 0})
        
        for entry in self.feedback_data:
            query_data[entry.query]["count"] += 1
            query_data[entry.query]["avg_score"] = (
                (query_data[entry.query]["avg_score"] * (query_data[entry.query]["count"] - 1) + 
                 entry.relevance_score) / query_data[entry.query]["count"]
            )
            
            if entry.relevance_score >= 0.5:
                query_data[entry.query]["positive"] += 1
            else:
                query_data[entry.query]["negative"] += 1
        
        # Sort by count and convert to list
        sorted_data = [
            {
                "query": query,
                "count": data["count"],
                "avg_score": data["avg_score"],
                "positive": data["positive"],
                "negative": data["negative"]
            }
            for query, data in sorted(
                query_data.items(), 
                key=lambda x: x[1]["count"], 
                reverse=True
            )
        ]
        
        return {"queries": sorted_data[:50]}  # Return top 50 queries
    
    def get_problematic_documents(self) -> List[Dict[str, Any]]:
        """Get documents with the most negative feedback.
        
        Returns:
            List of documents with negative feedback
        """
        if not self.feedback_data:
            return []
        
        # Group feedback by document
        doc_data = defaultdict(lambda: {"count": 0, "avg_score": 0, "text": "", "queries": set()})
        
        for entry in self.feedback_data:
            doc_id = entry.document_id
            doc_data[doc_id]["count"] += 1
            doc_data[doc_id]["avg_score"] = (
                (doc_data[doc_id]["avg_score"] * (doc_data[doc_id]["count"] - 1) + 
                 entry.relevance_score) / doc_data[doc_id]["count"]
            )
            doc_data[doc_id]["text"] = entry.document_text[:200] + "..."  # Store snippet
            doc_data[doc_id]["queries"].add(entry.query)
        
        # Find documents with negative scores and multiple feedback entries
        problematic_docs = [
            {
                "document_id": doc_id,
                "avg_score": data["avg_score"],
                "feedback_count": data["count"],
                "snippet": data["text"],
                "queries": list(data["queries"])
            }
            for doc_id, data in doc_data.items()
            if data["avg_score"] < 0.5 and data["count"] >= 3  # At least 3 feedback entries with bad score
        ]
        
        # Sort by avg_score (worst first)
        return sorted(problematic_docs, key=lambda x: x["avg_score"])
    
    def generate_analytics_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate a comprehensive analytics report.
        
        Args:
            output_path: Path to save the report visualization
            
        Returns:
            Dictionary with report data
        """
        if not self.feedback_data:
            return {"error": "No feedback data available"}
        
        # Get basic stats
        stats = self.get_feedback_stats()
        
        # Get timeline data
        timeline = self.get_feedback_timeline()
        
        # Get query analysis
        query_analysis = self.get_query_analysis()
        
        # Get problematic documents
        problematic_docs = self.get_problematic_documents()
        
        # Create visualizations if output path is provided
        if output_path:
            self._generate_visualizations(
                stats=stats,
                timeline=timeline,
                queries=query_analysis,
                output_path=output_path
            )
        
        # Compile report
        report = {
            "timestamp": datetime.now().isoformat(),
            "stats": stats,
            "timeline": timeline,
            "query_analysis": query_analysis,
            "problematic_documents": problematic_docs[:10]  # Top 10 worst
        }
        
        return report
    
    def _generate_visualizations(
        self,
        stats: Dict[str, Any],
        timeline: Dict[str, Any],
        queries: Dict[str, Any],
        output_path: str
    ):
        """Generate visualizations for the analytics report.
        
        Args:
            stats: Basic statistics
            timeline: Timeline data
            queries: Query analysis data
            output_path: Path to save visualizations
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up the figure style
        sns.set(style="whitegrid")
        plt.rcParams.update({'font.size': 12})
        
        # 1. Feedback Distribution Pie Chart
        plt.figure(figsize=(10, 6))
        plt.pie(
            [stats["positive_feedback"], stats["negative_feedback"]],
            labels=["Positive", "Negative"],
            autopct='%1.1f%%',
            colors=["#4CAF50", "#F44336"],
            startangle=90
        )
        plt.title("Feedback Distribution")
        plt.tight_layout()
        plt.savefig(output_dir / "feedback_distribution.png")
        plt.close()
        
        # 2. Feedback Timeline
        if timeline["timeline"]:
            data = pd.DataFrame(timeline["timeline"])
            plt.figure(figsize=(12, 6))
            
            plt.subplot(2, 1, 1)
            plt.plot(data["period"], data["count"], marker='o', linestyle='-', color="#2196F3")
            plt.title("Feedback Volume Over Time")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plt.subplot(2, 1, 2)
            plt.plot(data["period"], data["avg_score"], marker='o', linestyle='-', color="#FF9800")
            plt.title("Average Relevance Score Over Time")
            plt.xticks(rotation=45)
            plt.tight_layout(pad=3.0)
            
            plt.savefig(output_dir / "feedback_timeline.png", bbox_inches="tight")
            plt.close()
        
        # 3. Top Queries Bar Chart
        if queries["queries"]:
            top_queries = pd.DataFrame(queries["queries"][:10])  # Top 10
            plt.figure(figsize=(12, 6))
            
            # Plot count bars
            bars = plt.bar(
                top_queries["query"], 
                top_queries["count"],
                color="#3F51B5"
            )
            
            # Add average score as text on each bar
            for bar, score in zip(bars, top_queries["avg_score"]):
                plt.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5,
                    f'Score: {score:.2f}',
                    ha='center',
                    va='bottom',
                    rotation=0
                )
            
            plt.title("Top 10 Queries by Feedback Count")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(output_dir / "top_queries.png")
            plt.close()
    
    def trigger_model_finetune(self):
        """Trigger model fine-tuning based on collected feedback."""
        if not self.feedback_data or len(self.feedback_data) < self.config.min_feedback_threshold:
            logger.info(f"Not enough feedback data for fine-tuning. Need at least {self.config.min_feedback_threshold} entries.")
            return False
        
        logger.info("Starting model fine-tuning process based on feedback")
        
        try:
            # Create model fine-tuner and run the process
            fine_tuner = FeedbackModelFineTuner(self.config)
            result = fine_tuner.fine_tune_from_feedback(self.feedback_data)
            
            if result["success"]:
                logger.info(f"Model fine-tuning completed successfully. New model saved at: {result['model_path']}")
                return True
            else:
                logger.error(f"Model fine-tuning failed: {result['error']}")
                return False
                
        except Exception as e:
            logger.error(f"Error during model fine-tuning: {e}")
            return False


class FeedbackModelFineTuner:
    """Fine-tune embedding models based on feedback data."""
    
    def __init__(self, config: FeedbackAnalyticsConfig):
        """Initialize model fine-tuner.
        
        Args:
            config: Configuration for feedback analytics and fine-tuning
        """
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def fine_tune_from_feedback(self, feedback_data: List[FeedbackEntry]) -> Dict[str, Any]:
        """Fine-tune a model based on feedback data.
        
        Args:
            feedback_data: List of feedback entries
            
        Returns:
            Dictionary with fine-tuning results
        """
        try:
            # Filter feedback data to get high-confidence entries
            filtered_data = [
                entry for entry in feedback_data
                if abs(entry.relevance_score - 0.5) >= (self.config.min_feedback_confidence - 0.5)
            ]
            
            if len(filtered_data) < self.config.min_feedback_threshold:
                return {
                    "success": False,
                    "error": f"Not enough high-confidence feedback data for fine-tuning. Need at least {self.config.min_feedback_threshold} entries."
                }
            
            # Prepare training and evaluation datasets
            train_data, eval_data = self._prepare_datasets(filtered_data)
            
            if not train_data or not eval_data:
                return {
                    "success": False,
                    "error": "Failed to prepare training and evaluation datasets."
                }
            
            # Load base model
            model = SentenceTransformer("all-MiniLM-L6-v2")
            model.to(self.device)
            
            # Set up training
            train_batch_size = self.config.batch_size
            num_epochs = self.config.training_epochs
            
            # Create data loader
            train_dataloader = DataLoader(
                train_data,
                shuffle=True,
                batch_size=train_batch_size
            )
            
            # Create loss function
            train_loss = losses.CosineSimilarityLoss(model)
            
            # Create evaluator
            evaluator = InformationRetrievalEvaluator(
                eval_data["queries"],
                eval_data["corpus"],
                eval_data["relevant_docs"]
            )
            
            # Train the model
            warmup_steps = int(len(train_dataloader) * num_epochs * self.config.warmup_ratio)
            
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                evaluator=evaluator,
                epochs=num_epochs,
                warmup_steps=warmup_steps,
                optimizer_params={'lr': self.config.learning_rate},
                output_path=self.config.finetune_models_dir
            )
            
            # Save model with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f"{self.config.finetune_models_dir}/model_{timestamp}"
            model.save(model_path)
            
            # Evaluate on test set
            evaluation_result = evaluator(model)
            
            return {
                "success": True,
                "model_path": model_path,
                "evaluation": evaluation_result,
                "training_samples": len(train_data),
                "evaluation_samples": len(eval_data["queries"])
            }
            
        except Exception as e:
            logger.error(f"Error in fine-tuning process: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _prepare_datasets(self, feedback_data: List[FeedbackEntry]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Prepare training and evaluation datasets from feedback.
        
        Args:
            feedback_data: List of feedback entries
            
        Returns:
            Tuple of (train_data, eval_data)
        """
        # Create corpus from all documents
        corpus = {}
        for entry in feedback_data:
            corpus[entry.document_id] = entry.document_text
        
        # Group by query
        query_groups = defaultdict(list)
        for entry in feedback_data:
            query_groups[entry.query].append(entry)
        
        # Create triplets for training
        train_data = []
        eval_queries = {}
        eval_relevant_docs = {}
        
        for query, entries in query_groups.items():
            # Sort by relevance score
            sorted_entries = sorted(entries, key=lambda x: x.relevance_score, reverse=True)
            
            if len(sorted_entries) < 2:
                continue
            
            # Split for train/eval
            if np.random.random() < self.config.evaluation_split:
                # Use for evaluation
                eval_queries[query] = query
                eval_relevant_docs[query] = set()
                
                for entry in sorted_entries:
                    if entry.relevance_score >= 0.5:
                        eval_relevant_docs[query].add(entry.document_id)
            else:
                # Use for training
                positive_docs = [e.document_id for e in sorted_entries if e.relevance_score >= 0.75]
                negative_docs = [e.document_id for e in sorted_entries if e.relevance_score <= 0.25]
                
                if positive_docs and negative_docs:
                    for pos_id in positive_docs:
                        for neg_id in negative_docs:
                            train_data.append({
                                "query": query,
                                "pos": corpus[pos_id],
                                "neg": corpus[neg_id]
                            })
        
        # Limit training data if needed
        if len(train_data) > self.config.max_training_samples:
            train_data = np.random.choice(
                train_data, 
                size=self.config.max_training_samples, 
                replace=False
            ).tolist()
        
        eval_data = {
            "queries": eval_queries,
            "corpus": corpus,
            "relevant_docs": eval_relevant_docs
        }
        
        return train_data, eval_data
