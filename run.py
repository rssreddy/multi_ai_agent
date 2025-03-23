#!/usr/bin/env python3
"""
Multi-AI Agent Workflow System
Main entry point for running the application
"""

import argparse
from pprint import pprint
from core.workflow import WorkflowManager
from utils.logger import logger

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Multi-AI Agent Workflow System')
    parser.add_argument('--query', '-q', type=str, help='User query to process')
    parser.add_argument('--interactive', '-i', action='store_true', help='Run in interactive mode')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    return parser.parse_args()

def process_query(query: str, verbose: bool = False):
    """
    Process a single query through the workflow.
    
    Args:
        query: The user query to process
        verbose: Whether to show verbose output
    """
    # Create and build the workflow
    workflow = WorkflowManager().build_graph()
    
    # Process the query
    print(f"\nProcessing query: '{query}'")
    print("-" * 50)
    
    if verbose:
        # Stream the workflow execution for verbose output
        for output in workflow.stream(query):
            for key, value in output.items():
                if value is None:
                    continue
                # Get the last message from the output
                last_message = value.get("messages", [])[-1] if "messages" in value else None
                if last_message:
                    print(f"\nOutput from node '{key}':")
                    pprint(last_message, indent=2, width=80, depth=None)
                    print()
    else:
        # Run the workflow and get the final result
        result = workflow.run(query)
        # Extract the final answer
        final_answer = result["messages"][-2].content  # -2 because -1 is validator's reason
        print("\nFinal Answer:")
        print("-" * 50)
        print(final_answer)
        print("-" * 50)

def interactive_mode():
    """Run the application in interactive mode."""
    print("Multi-AI Agent Workflow System - Interactive Mode")
    print("Type 'exit' or 'quit' to end the session")
    print("-" * 50)
    
    while True:
        query = input("\nEnter your query: ")
        if query.lower() in ('exit', 'quit'):
            break
        
        if not query.strip():
            continue
        
        try:
            process_query(query)
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            print(f"An error occurred: {e}")

def main():
    """Main entry point for the application."""
    args = parse_arguments()
    
    if args.interactive:
        interactive_mode()
    elif args.query:
        process_query(args.query, args.verbose)
    else:
        print("Please provide a query with --query or use --interactive mode")
        print("Example: python run.py --query 'What is the difference between the stock price of Infosys in 2023 and 2021?'")
        print("Example: python run.py --interactive")

if __name__ == "__main__":
    main()