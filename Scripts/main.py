from process_documents import QdrantManager

def main():
    """Main execution flow"""
    logger.info(f"Starting document processing - {SystemConfig.CURRENT_TIME}")
    logger.info(f"User: {SystemConfig.CURRENT_USER}")
    
    try:
        # Initialize components
        document_processor = DocumentProcessor()
        qdrant_manager = QdrantManager()
        performance_monitor = PerformanceMonitor()
        
        # Start overall performance monitoring
        performance_monitor.start()
        
        # Create Qdrant collection
        qdrant_manager.create_collection()
        
        # Get files to process
        documents_dir = "Documents"
        if not os.path.exists(documents_dir):
            os.makedirs(documents_dir, exist_ok=True)
            logger.info(f"Created '{documents_dir}' directory. Please add documents and run again.")
            return
            
        # Load processed files log
        processed_files = load_processed_files()
        
        # Get new files to process
        supported_extensions = (
            '.pdf', '.docx', '.txt', '.xlsx', '.csv', '.sql', '.parquet',
            '.json', '.xml', '.html', '.htm', '.yaml', '.yml', '.md',
            '.pptx', '.rtf'
        )
        
        new_files = []
        for root, _, files in os.walk(documents_dir):
            for file in files:
                if file.lower().endswith(supported_extensions):
                    full_path = os.path.join(root, file)
                    if document_processor.needs_processing(full_path, processed_files):
                        new_files.append(full_path)
                        
        if not new_files:
            logger.info("No new documents to process")
            return
            
        logger.info(f"Found {len(new_files)} new documents to process")
        
        # Process documents
        all_chunks = document_processor.process_documents_parallel(new_files)
        
        # Add to Qdrant
        qdrant_manager.add_chunks_to_qdrant(all_chunks)
        
        # Update processed files log
        document_processor.update_processed_files(new_files, processed_files)
        save_processed_files(processed_files)
        
        # Log final metrics
        metrics = performance_monitor.stop()
        logger.info(f"Processing completed. Final metrics: {metrics}")
        logger.info(f"Successfully processed {len(new_files)} documents")
        logger.info(f"Added {len(all_chunks)} chunks to Qdrant")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        traceback.print_exc()
        
if __name__ == "__main__":
    main()