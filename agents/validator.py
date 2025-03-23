from typing import Literal
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, END
from langgraph.types import Command

from agents.base import BaseAgent
from core.models import Validator
from core.state import WorkflowState
from config.settings import VALIDATOR_PROMPT
from utils.logger import logger

class ValidatorAgent(BaseAgent):
    """
    Validator agent that ensures the quality of the workflow output.
    Determines whether to end the workflow or continue processing.
    """
    
    def process(self, state: MessagesState) -> Command[Literal["supervisor", "__end__"]]:
        """
        Process the current state to validate the quality of the response.
        
        Args:
            state: The current workflow state
            
        Returns:
            A Command object routing to either the supervisor or end
        """
        # Extract user question and agent answer
        user_question = WorkflowState.get_user_question(state)
        agent_answer = WorkflowState.get_last_response(state)
        
        # Prepare messages for validation
        messages = [
            {"role": "system", "content": VALIDATOR_PROMPT},
            {"role": "user", "content": user_question},
            {"role": "assistant", "content": agent_answer},
        ]
        
        # Get structured output from the LLM
        response = self.llm.with_structured_output(Validator).invoke(messages)
        
        # Extract routing decision and reason
        goto = response.next
        reason = response.reason
        
        # Determine the next node
        if goto == "FINISH" or goto == END:
            goto = END
            logger.info("Transitioning to END")
        else:
            self.log_transition("supervisor")
        
        # Return command with updated state and next destination
        return Command(
            update={
                "messages": [
                    HumanMessage(content=reason, name="validator")
                ]
            },
            goto=goto
        )