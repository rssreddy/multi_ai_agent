from abc import ABC, abstractmethod
from typing import Literal, Any
from langchain_groq import ChatGroq
from langgraph.graph import MessagesState
from langgraph.types import Command
from config.settings import GROQ_API_KEY, LLM_MODEL
from utils.logger import logger

class BaseAgent(ABC):
    """
    Abstract base class for all agents in the workflow.
    Defines the common interface and shared functionality.
    """
    
    def __init__(self):
        """Initialize the agent with a language model."""
        self.llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=LLM_MODEL)
        self.name = self.__class__.__name__.lower().replace('agent', '')
    
    @abstractmethod
    def process(self, state: MessagesState) -> Command:
        """
        Process the current state and return a command for the next step.
        
        Args:
            state: The current workflow state
            
        Returns:
            A Command object indicating the next step
        """
        pass
    
    def log_transition(self, next_node: str):
        """
        Log the transition from this agent to the next node.
        
        Args:
            next_node: The name of the next node
        """
        logger.node_transition(self.name, next_node)
    
    def prepare_messages(self, system_prompt: str, state: MessagesState) -> list:
        """
        Prepare messages for the language model by combining system prompt with state.
        
        Args:
            system_prompt: The system prompt to guide the LLM
            state: The current workflow state
            
        Returns:
            A list of messages ready for the LLM
        """
        return [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]