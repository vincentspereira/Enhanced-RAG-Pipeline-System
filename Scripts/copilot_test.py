import requests
from datetime import datetime
from typing import Dict, List, Tuple

class CopilotRAGTester:
    def __init__(self):
        self.api_url = "http://localhost:8000/query"
        self.current_user = "vincentspereira"
        self.current_time = "2025-05-21 02:19:15"
        
    def get_context(self, query: str, top_k: int = 3) -> Tuple[str, List[Dict]]:
        """Get context from RAG system"""
        try:
            response = requests.post(
                self.api_url,
                json={"query": query, "top_k": top_k}
            )
            results = response.json()
            
            # Format context for Copilot
            context = "Context from document repository:\n\n"
            for i, result in enumerate(results, 1):
                context += f"[Document {i}] "
                context += f"(Source: {result.get('metadata', {}).get('source', 'Unknown')})\n"
                context += f"{result.get('text', '')}\n\n"
                
            return context, results
        except Exception as e:
            print(f"Error getting context: {e}")
            return "", []
            
    def generate_copilot_prompt(self, query: str, context: str) -> str:
        """Generate a well-formatted prompt for GitHub Copilot"""
        prompt = f"I have the following information from my document repository:\n\n"
        prompt += context
        prompt += "\n---\n\n"
        prompt += f"Based on this information, please help me with: {query}\n"
        return prompt

if __name__ == "__main__":
    tester = CopilotRAGTester()
    
    print("GitHub Copilot RAG Integration Tester")
    print(f"User: {tester.current_user}")
    print(f"Time: {tester.current_time}")
    print("-" * 50)
    
    while True:
        query = input("\nEnter your question (or 'quit' to exit): ").strip()
        if query.lower() == 'quit':
            break
            
        context, results = tester.get_context(query)
        prompt = tester.generate_copilot_prompt(query, context)
        
        print("\nCopy the following prompt into your GitHub Copilot chat:")
        print("-" * 50)
        print(prompt)
        print("-" * 50)