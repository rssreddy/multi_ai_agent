# Multi-AI Agent System

A modular, extensible system for coordinating multiple specialized AI agents in a workflow.

## Project Structure

```
multi_ai_agent/
├── config/
│   ├── __init__.py
│   └── settings.py          # Configuration management
├── core/
│   ├── __init__.py
│   ├── base.py              # Abstract base classes
│   ├── messages.py          # Message handling
│   ├── registry.py          # Agent and tool registry
│   └── workflow.py          # Workflow management
├── agents/
│   ├── __init__.py
│   ├── supervisor.py        # Supervisor agent
│   ├── enhancer.py          # Prompt enhancer agent
│   ├── researcher.py        # Research agent
│   ├── coder.py             # Code execution agent
│   └── validator.py         # Validation agent
├── tools/
│   ├── __init__.py
│   ├── search.py            # Search tools
│   └── code_execution.py    # Code execution tools
├── models/
│   ├── __init__.py
│   └── schemas.py           # Pydantic models for structured outputs
├── factories/
│   ├── __init__.py
│   ├── agent_factory.py     # Agent creation factory
│   └── tool_factory.py      # Tool creation factory
└── main.py                  # Application entry point
```

## Key Design Patterns and SOLID Principles

1. **Single Responsibility Principle**: Each class has one responsibility
2. **Open/Closed Principle**: Extensible through inheritance rather than modification
3. **Liskov Substitution Principle**: Agents and tools follow consistent interfaces
4. **Interface Segregation**: Clean, focused interfaces
5. **Dependency Inversion**: High-level modules depend on abstractions

## Design Patterns Used

- **Factory Pattern**: For creating agents and tools
- **Strategy Pattern**: For interchangeable algorithms in agents
- **Registry Pattern**: For managing available components
- **Command Pattern**: For encapsulating operations

## How to Run

### Prerequisites

1. Make sure you have Python 3.x installed
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up your environment variables (API keys for LLM providers, search tools, etc.) in a `.env` file

### Running the Application

You can run the application in two ways:

#### 1. Single Query Mode

Process a single query:

```bash
python run.py --query "Your question or task here"
```

For detailed output showing each step of the workflow:

```bash
python run.py --query "Your question or task here" --verbose
```

#### 2. Interactive Mode

Start an interactive session where you can enter multiple queries:

```bash
python run.py --interactive
```

### Example

```bash
# Process a specific query
python run.py --query "What are the latest developments in quantum computing?"

# Run in interactive mode
python run.py --interactive
```

In interactive mode, type 'exit' or 'quit' to end the session.
