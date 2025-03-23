from langgraph.graph import MessagesState

class WorkflowState:
    """
    Manages the state of the workflow, providing a consistent interface
    for state operations across different components.
    """
    
    @staticmethod
    def create_initial_state(user_query: str) -> dict:
        """
        Creates the initial state for the workflow with a user query.
        
        Args:
            user_query: The initial query from the user
            
        Returns:
            A dictionary with the initial state
        """
        return {
            "messages": [
                ("user", user_query),
            ]
        }
    
    @staticmethod
    def get_user_question(state: MessagesState) -> str:
        """
        Extracts the user's original question from the state.
        
        Args:
            state: The current workflow state
            
        Returns:
            The user's original question as a string
        """
        return state["messages"][0].content
    
    @staticmethod
    def get_last_response(state: MessagesState) -> str:
        """
        Extracts the last response from the state.
        
        Args:
            state: The current workflow state
            
        Returns:
            The last response in the state as a string
        """
        return state["messages"][-1].content