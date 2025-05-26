#!/usr/bin/env python
"""
CLI tool for managing and analyzing feedback data.

This tool allows you to generate analytics reports, visualize feedback trends,
and trigger model fine-tuning based on collected user feedback.
"""
import argparse
import logging
import sys
import json
from pathlib import Path
from datetime import datetime

from Scripts.enhancers.feedback_analytics import FeedbackAnalytics, FeedbackAnalyticsConfig, FeedbackEntry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_report(args):
    """Generate feedback analytics report."""
    config = FeedbackAnalyticsConfig(
        storage_path=args.storage_path,
        aggregation_window=args.aggregation
    )
    
    analytics = FeedbackAnalytics(config)
    
    output_path = args.output_path or f"reports/feedback_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report = analytics.generate_analytics_report(output_path)
    
    # Save report as JSON
    report_file = Path(output_path) / "report.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Report generated successfully at {output_path}")
    logger.info(f"Total feedback entries: {report['stats']['total_feedback']}")
    logger.info(f"Positive feedback: {report['stats']['positive_feedback']} ({report['stats']['positive_feedback']/report['stats']['total_feedback']*100:.1f}%)")
    logger.info(f"Average relevance score: {report['stats']['average_relevance']:.2f}")

def finetune_model(args):
    """Trigger model fine-tuning based on feedback."""
    config = FeedbackAnalyticsConfig(
        storage_path=args.storage_path,
        min_feedback_threshold=args.min_feedback,
        training_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        finetune_models_dir=args.models_dir
    )
    
    analytics = FeedbackAnalytics(config)
    
    logger.info("Triggering model fine-tuning based on feedback data")
    result = analytics.trigger_model_finetune()
    
    if result:
        logger.info("Model fine-tuning completed successfully")
    else:
        logger.error("Model fine-tuning failed")

def add_feedback(args):
    """Add feedback entry manually."""
    config = FeedbackAnalyticsConfig(storage_path=args.storage_path)
    analytics = FeedbackAnalytics(config)
    
    # Create feedback entry
    feedback = FeedbackEntry(
        query=args.query,
        document_id=args.document_id,
        document_text=args.text,
        relevance_score=args.score,
        user_id=args.user_id
    )
    
    # Add feedback
    analytics.add_feedback(feedback)
    logger.info(f"Feedback added successfully for query: {args.query}")

def import_feedback(args):
    """Import feedback from JSON file."""
    config = FeedbackAnalyticsConfig(storage_path=args.storage_path)
    analytics = FeedbackAnalytics(config)
    
    import_file = Path(args.import_file)
    if not import_file.exists():
        logger.error(f"Import file not found: {args.import_file}")
        return
    
    try:
        with open(import_file, "r") as f:
            feedback_data = json.load(f)
        
        count = 0
        for entry in feedback_data:
            feedback = FeedbackEntry(
                query=entry["query"],
                document_id=entry["document_id"],
                document_text=entry["document_text"],
                relevance_score=float(entry["relevance_score"]),
                timestamp=entry.get("timestamp", datetime.now().isoformat()),
                user_id=entry.get("user_id"),
                session_id=entry.get("session_id"),
                metadata=entry.get("metadata", {})
            )
            
            analytics.add_feedback(feedback)
            count += 1
        
        logger.info(f"Successfully imported {count} feedback entries")
    
    except Exception as e:
        logger.error(f"Error importing feedback: {e}")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Feedback Analytics CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Report generation command
    report_parser = subparsers.add_parser("report", help="Generate feedback analytics report")
    report_parser.add_argument("--storage-path", default="data/feedback", help="Path to feedback storage")
    report_parser.add_argument("--output-path", help="Path to save report and visualizations")
    report_parser.add_argument("--aggregation", choices=["hour", "day", "week", "month"], default="day", 
                             help="Time window for data aggregation")
    
    # Model fine-tuning command
    finetune_parser = subparsers.add_parser("finetune", help="Trigger model fine-tuning")
    finetune_parser.add_argument("--storage-path", default="data/feedback", help="Path to feedback storage")
    finetune_parser.add_argument("--min-feedback", type=int, default=100, help="Minimum feedback entries required")
    finetune_parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    finetune_parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    finetune_parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    finetune_parser.add_argument("--models-dir", default="data/models/finetuned", help="Directory for saving fine-tuned models")
    
    # Add feedback command
    add_parser = subparsers.add_parser("add", help="Add feedback entry")
    add_parser.add_argument("--storage-path", default="data/feedback", help="Path to feedback storage")
    add_parser.add_argument("--query", required=True, help="Search query")
    add_parser.add_argument("--document-id", required=True, help="Document ID")
    add_parser.add_argument("--text", required=True, help="Document text")
    add_parser.add_argument("--score", type=float, required=True, help="Relevance score (0.0-1.0)")
    add_parser.add_argument("--user-id", help="User ID")
    
    # Import feedback command
    import_parser = subparsers.add_parser("import", help="Import feedback from JSON file")
    import_parser.add_argument("--storage-path", default="data/feedback", help="Path to feedback storage")
    import_parser.add_argument("--import-file", required=True, help="JSON file with feedback data")
    
    args = parser.parse_args()
    
    if args.command == "report":
        generate_report(args)
    elif args.command == "finetune":
        finetune_model(args)
    elif args.command == "add":
        add_feedback(args)
    elif args.command == "import":
        import_feedback(args)
    else:
        parser.print_help()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
