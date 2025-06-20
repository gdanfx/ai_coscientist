"""
AI Co-Scientist Agents

This package contains all the specialized agents that work together to conduct
automated scientific research through hypothesis generation, evaluation, ranking,
and evolution.
"""

# Import agents with error handling for optional dependencies
try:
    from .generation import EnhancedGenerationAgent
    GENERATION_AVAILABLE = True
except ImportError:
    GENERATION_AVAILABLE = False

try:
    from .proximity import ProximityAgent, run_proximity_agent
    PROXIMITY_AVAILABLE = True
except ImportError:
    PROXIMITY_AVAILABLE = False

try:
    from .reflection import RobustReflectionAgent, run_reflection_agent, create_reflection_agent
    REFLECTION_AVAILABLE = True
except ImportError:
    REFLECTION_AVAILABLE = False

try:
    from .ranking import RankingAgent, EloSystem, run_ranking_agent, create_ranking_agent
    RANKING_AVAILABLE = True
except ImportError:
    RANKING_AVAILABLE = False

try:
    from .evolution import EvolutionAgent, run_evolution_agent, create_evolution_agent
    EVOLUTION_AVAILABLE = True
except ImportError:
    EVOLUTION_AVAILABLE = False

try:
    from .supervisor import IntegratedSupervisor, SupervisorFactory
    SUPERVISOR_AVAILABLE = True
except ImportError:
    SUPERVISOR_AVAILABLE = False

__all__ = []

# Add conditional exports based on available agents
if GENERATION_AVAILABLE:
    __all__.append('EnhancedGenerationAgent')

if PROXIMITY_AVAILABLE:
    __all__.extend(['ProximityAgent', 'run_proximity_agent'])

if REFLECTION_AVAILABLE:
    __all__.extend(['RobustReflectionAgent', 'run_reflection_agent', 'create_reflection_agent'])

if RANKING_AVAILABLE:
    __all__.extend(['RankingAgent', 'EloSystem', 'run_ranking_agent', 'create_ranking_agent'])

if EVOLUTION_AVAILABLE:
    __all__.extend(['EvolutionAgent', 'run_evolution_agent', 'create_evolution_agent'])

if SUPERVISOR_AVAILABLE:
    __all__.extend(['IntegratedSupervisor', 'SupervisorFactory'])