from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.riza.command import ExecPython
from config.settings import TAVILY_MAX_RESULTS

class ToolFactory:
    """
    Factory class for creating and managing tools used by agents.
    Follows the Factory pattern to centralize tool creation.
    """
    
    @staticmethod
    def create_tavily_search() -> TavilySearchResults:
        """
        Creates a TavilySearchResults tool instance.
        
        Returns:
            Configured TavilySearchResults tool
        """
        return TavilySearchResults(max_results=TAVILY_MAX_RESULTS)
    
    @staticmethod
    def create_python_executor() -> ExecPython:
        """
        Creates an ExecPython tool instance.
        
        Returns:
            Configured ExecPython tool
        """
        return ExecPython()
    
    @classmethod
    def create_all_tools(cls) -> list:
        """
        Creates all available tools.
        
        Returns:
            List of all tool instances
        """
        return [
            cls.create_tavily_search(),
            cls.create_python_executor()
        ]
    
    @classmethod
    def create_research_tools(cls) -> list:
        """
        Creates tools specifically for research tasks.
        
        Returns:
            List of research-focused tool instances
        """
        return [cls.create_tavily_search()]
    
    @classmethod
    def create_coding_tools(cls) -> list:
        """
        Creates tools specifically for coding tasks.
        
        Returns:
            List of coding-focused tool instances
        """
        return [cls.create_python_executor()]