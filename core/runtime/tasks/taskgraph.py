from typing import Dict, List
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from core.utils.logger import get_logger
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

logger = get_logger(__name__)

class TaskGraphBuilder:
    def __init__(self):
        # Check if OpenAI API key is set
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is not set")
            
        self.llm = ChatOpenAI(
            model="gpt-4",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
    def build_graph(self, task_description: str) -> str:
        """Build a task graph (DAG) from a task description using a single LLM query.
        
        Args:
            task_description: The description of the task to build a graph for
            
        Returns:
            A string representation of the task graph
        """
        try:
            messages = [
                HumanMessage(content=f"""Given the following task description, create a Directed Acyclic Graph (DAG) representation:

                {task_description}

                Please analyze the task and create a DAG where:
                1. Each node represents a distinct step or subtask
                2. Edges represent dependencies between tasks
                3. The graph should be acyclic (no circular dependencies)
                4. Each node should have a clear description of what it does
                5. The graph should show the complete execution flow

                Return the graph in a tree-like string format, for example:
                └── Root Task
                    ├── Subtask 1
                    │   └── Subtask 1.1
                    └── Subtask 2
                        └── Subtask 2.1

                For each node, include a brief description of what it does.
                """)
            ]
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Error building task graph: {str(e)}")
            return f"Error building task graph: {str(e)}" 