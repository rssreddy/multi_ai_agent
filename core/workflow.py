from typing import Dict, Any, Generator
from langgraph.graph import StateGraph, START, END, MessagesState

from agents import (
    SupervisorAgent,
    EnhancerAgent,
    ResearcherAgent,
    CoderAgent,
    ValidatorAgent
)
from core.state import WorkflowState
from utils.logger import logger

class WorkflowManager:
    """
    Manages the workflow graph construction and execution.
    Implements the Builder pattern for constructing the workflow graph.
    """
    
    def __init__(self):
        """Initialize the workflow manager with agent instances."""
        self.supervisor = SupervisorAgent()
        self.enhancer = EnhancerAgent()
        self.researcher = ResearcherAgent()
        self.coder = CoderAgent()
        self.validator = ValidatorAgent()
        self.graph = None
    
    def build_graph(self) -> 'WorkflowManager':
        """
        Build the workflow graph with all nodes and edges.
        
        Returns:
            Self for method chaining
        """
        # Initialize the graph builder
        builder = StateGraph(MessagesState)
        
        # Add nodes to the graph
        builder.add_node("supervisor", self.supervisor.process)
        builder.add_node("enhancer", self.enhancer.process)
        builder.add_node("researcher", self.researcher.process)
        builder.add_node("coder", self.coder.process)
        builder.add_node("validator", self.validator.process)
        
        # Add edges to define the workflow
        builder.add_edge(START, "supervisor")
        
        # Compile the graph
        self.graph = builder.compile()
        
        return self
    
    def run(self, user_query: str) -> Dict[str, Any]:
        """
        Run the workflow with a user query and return the final result.
        
        Args:
            user_query: The user's query to process
            
        Returns:
            The final state after workflow completion
        """
        if not self.graph:
            self.build_graph()
        
        # Create initial state with user query
        initial_state = WorkflowState.create_initial_state(user_query)
        
        # Execute the workflow
        logger.info(f"Starting workflow with query: {user_query}")
        result = self.graph.invoke(initial_state)
        
        return result
    
    def stream(self, user_query: str) -> Generator[Dict[str, Any], None, None]:
        """
        Stream the workflow execution with a user query.
        
        Args:
            user_query: The user's query to process
            
        Yields:
            Intermediate states during workflow execution
        """
        if not self.graph:
            self.build_graph()
        
        # Create initial state with user query
        initial_state = WorkflowState.create_initial_state(user_query)
        
        # Stream the workflow execution
        logger.info(f"Starting workflow stream with query: {user_query}")
        yield from self.graph.stream(initial_state)