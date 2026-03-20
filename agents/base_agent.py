"""Base agent class for LangGraph multi-agent system."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import time
import logging

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Agent execution status"""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING = "waiting"


@dataclass
class AgentMetrics:
    """Metrics collected during agent execution"""
    execution_time: float = 0.0
    status: AgentStatus = AgentStatus.IDLE
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_time": self.execution_time,
            "status": self.status.value,
            "error_message": self.error_message,
            "warnings": self.warnings,
            **self.custom_metrics,
        }


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the multi-agent system.

    Each agent follows a simple lifecycle:
    1. Initialize with configuration
    2. Process input state and produce output state
    3. Return updated state with metrics
    """

    def __init__(self, config: Dict[str, Any], agent_name: Optional[str] = None):
        """
        Initialize the agent with configuration.

        Args:
            config: Agent-specific configuration
            agent_name: Optional name for the agent (defaults to class name)
        """
        self.config = config
        self.name = agent_name or self.__class__.__name__
        self.metrics = AgentMetrics()
        self._initialize()

    @abstractmethod
    def _initialize(self) -> None:
        """
        Initialize agent-specific resources.
        Override this in subclasses to set up models, load data, etc.
        """
        pass

    @abstractmethod
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input state and produce output state.

        Args:
            state: Current state dictionary from LangGraph

        Returns:
            Updated state dictionary with agent's outputs
        """
        pass

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent with timing and error handling.

        Args:
            state: Input state

        Returns:
            Updated state with outputs and metrics
        """
        start_time = time.time()
        self.metrics.status = AgentStatus.RUNNING

        try:
            logger.info(f"[{self.name}] Starting processing")
            result = self.process(state)

            # Add agent name to outputs
            result[f"{self.name}_output"] = True
            result[f"{self.name}_metrics"] = self.metrics.to_dict()

            self.metrics.status = AgentStatus.COMPLETED
            logger.info(f"[{self.name}] Completed successfully")

            return result

        except Exception as e:
            self.metrics.status = AgentStatus.FAILED
            self.metrics.error_message = str(e)
            logger.error(f"[{self.name}] Failed: {e}", exc_info=True)

            # Return state with error info
            state["errors"] = state.get("errors", [])
            state["errors"].append({
                "agent": self.name,
                "error": str(e),
                "timestamp": time.time(),
            })
            return state

        finally:
            self.metrics.execution_time = time.time() - start_time

    def validate_input(self, state: Dict[str, Any], required_keys: List[str]) -> bool:
        """
        Validate that required keys exist in the state.

        Args:
            state: State dictionary to validate
            required_keys: List of required key names

        Returns:
            True if all keys present, False otherwise
        """
        missing = [key for key in required_keys if key not in state]
        if missing:
            self.metrics.warnings.append(f"Missing required keys: {missing}")
            return False
        return True

    def get_metrics(self) -> Dict[str, Any]:
        """Get current agent metrics"""
        return self.metrics.to_dict()

    def reset(self) -> None:
        """Reset agent state for reuse"""
        self.metrics = AgentMetrics()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
