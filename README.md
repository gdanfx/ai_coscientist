# AI Co-Scientist: Autonomous Scientific Research System

A sophisticated multi-agent AI research pipeline that autonomously conducts complete scientific research cycles: literature review → hypothesis generation → peer review → competitive ranking → creative improvement → meta-analysis.

## 🎯 Overview

The AI Co-Scientist system implements a cutting-edge multi-agent architecture that mimics the complete scientific research process. It orchestrates specialized AI agents working together to:

- **Search and analyze** academic literature from multiple sources
- **Generate novel hypotheses** through intelligent synthesis
- **Conduct automated peer review** with structured evaluation criteria
- **Rank hypotheses competitively** using Elo tournament systems
- **Evolve and improve** ideas through creative strategies
- **Detect patterns and biases** through meta-analysis

## 🏗️ Architecture

### Multi-Agent Pipeline

```
Generation Agent → Proximity Agent → Reflection Agent → Ranking Agent → Evolution Agent → Meta-Review Agent
                                                ↑                                           ↓
                                        Supervisor Agent ←—————————————————————————————————————
```

### Core Components

- **🔍 Generation Agent**: Multi-source literature search (PubMed, ArXiv, CrossRef) + hypothesis generation
- **🎯 Proximity Agent**: Semantic deduplication using sentence transformers and clustering  
- **📝 Reflection Agent**: Automated peer review with 5 criteria (novelty, feasibility, rigor, impact, testability)
- **🏆 Ranking Agent**: Elo tournament system for competitive hypothesis ranking via LLM debates
- **🧬 Evolution Agent**: Multi-strategy hypothesis improvement (simplification, combination, analogical reasoning)
- **📊 Meta-Review Agent**: System-level pattern analysis and bias detection
- **🎛️ Supervisor Agent**: Central orchestrator managing complete research cycles

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Google AI API key (for Gemini LLM)
- Email address (for PubMed API access)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai_coscientist
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and email
   ```

4. **Set required environment variables**
   ```bash
   export GOOGLE_API_KEY="your_gemini_api_key_here"
   export PUBMED_EMAIL="your_email@example.com"
   ```

### Basic Usage

#### Command Line Interface

```bash
# Run with default settings
python main.py

# Specify research goal
python main.py --research-goal "AI applications in healthcare"

# Advanced configuration
python main.py --research-goal "Drug discovery using ML" \
               --max-cycles 5 \
               --evolution-every 2 \
               --constraints "Focus on FDA-approved compounds"
```

#### Programmatic Usage

```python
from agents.supervisor import IntegratedSupervisor
from core.data_structures import SupervisorConfig

# Configure research parameters
config = SupervisorConfig(
    research_goal="Novel approaches to cancer immunotherapy",
    constraints=["Consider ethical implications", "Focus on patient safety"],
    max_cycles=3,
    evolution_every=2,
    meta_every=2
)

# Run complete research cycle
supervisor = IntegratedSupervisor(config)
results = supervisor.run()

# Access results
print(f"Generated {len(results.final_hypotheses)} unique hypotheses")
print(f"Top hypothesis: {results.final_hypotheses[0]['content']}")
```

## 📋 Configuration

### Environment Variables

| Variable | Description | Required | Example |
|----------|-------------|----------|---------|
| `GOOGLE_API_KEY` | Google AI API key for Gemini LLM | Yes | `AIza...` |
| `PUBMED_EMAIL` | Email for PubMed API access | Yes | `researcher@university.edu` |
| `CROSSREF_EMAIL` | Email for CrossRef API access | No | `researcher@university.edu` |
| `LOG_LEVEL` | Logging level | No | `INFO` |
| `USE_MOCK_LLM` | Use mock LLM for testing | No | `false` |

### Research Configuration

```python
config = SupervisorConfig(
    research_goal="Your research question",
    constraints=["Constraint 1", "Constraint 2"],
    max_cycles=5,           # Maximum research cycles
    evolution_every=2,      # Evolve hypotheses every N cycles
    meta_every=2,          # Run meta-analysis every N cycles
    proximity_every=1,      # Run deduplication every N cycles
    no_improve_patience=3   # Stop after N cycles without improvement
)
```

## 🔬 Research Workflow

### 1. Literature Search & Hypothesis Generation

The system searches multiple academic databases and synthesizes findings into novel hypotheses:

```python
# Individual agent usage
from agents.generation import EnhancedGenerationAgent

agent = EnhancedGenerationAgent()
result = agent.run_complete_workflow(
    research_goal="AI in drug discovery",
    constraints=["Focus on machine learning approaches"]
)
```

### 2. Semantic Deduplication

Remove duplicate or highly similar hypotheses while maintaining intellectual diversity:

```python
from agents.proximity import ProximityAgent

agent = ProximityAgent(distance_threshold=0.25)
result = agent.run(hypotheses)
print(f"Reduced from {len(hypotheses)} to {len(result.unique_hypotheses)} unique hypotheses")
```

### 3. Automated Peer Review

Evaluate hypotheses across multiple scientific criteria:

```python
from agents.reflection import RobustReflectionAgent

agent = RobustReflectionAgent()
result = agent.run(hypotheses)
for review in result.hypothesis_reviews:
    print(f"Hypothesis {review.hypothesis_id}: {review.overall_score}/10")
```

### 4. Competitive Ranking

Rank hypotheses through pairwise LLM debates using Elo ratings:

```python
from agents.ranking import RankingAgent

agent = RankingAgent()
result = agent.run(reviewed_hypotheses)
print("Final Rankings:")
for ranking in result.final_rankings:
    print(f"Rank {ranking.final_rank}: {ranking.hypothesis_content}")
```

### 5. Hypothesis Evolution

Improve top hypotheses using various creative strategies:

```python
from agents.evolution import EvolutionAgent

agent = EvolutionAgent()
result = agent.run(top_hypotheses)
for evolved in result.evolved_hypotheses:
    print(f"Original: {evolved.original_content}")
    print(f"Evolved: {evolved.evolved_content}")
```

### 6. Meta-Analysis

Analyze patterns and biases in the evaluation process:

```python
from agents.meta_review import EnhancedMetaReviewAgent

agent = EnhancedMetaReviewAgent()
result = agent.run(reflection_state)
print(f"Detected {len(result.pattern_insights)} evaluation patterns")
print(f"Recommendations: {result.actionable_for_generation}")
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python run_tests.py

# Run specific test suites
python -m pytest tests/unit/ -v
python -m pytest tests/integration/ -v

# Run individual agent tests
python -m pytest tests/unit/test_generation.py -v
python -m pytest tests/unit/test_proximity.py -v
python -m pytest tests/unit/test_reflection.py -v
```

### Test Categories

- **Unit Tests**: Test individual agent functionality
- **Integration Tests**: Test agent interactions and data flow
- **System Tests**: End-to-end workflow validation

### Running Individual Agents

```bash
# Test individual agents
python -m agents.generation
python -m agents.proximity  
python -m agents.reflection
python -m agents.ranking
python -m agents.evolution
python -m agents.meta_review
```

## 📁 Project Structure

```
ai_coscientist/
├── .env.example              # Environment template
├── .gitignore               # Git ignore rules  
├── requirements.txt         # Python dependencies
├── main.py                  # CLI entry point
├── CLAUDE.md               # Development guidance
├── README.md               # This file
│
├── core/                   # Core system components
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   └── data_structures.py  # Shared data structures
│
├── agents/                 # AI agent implementations
│   ├── __init__.py
│   ├── generation.py       # Literature search & hypothesis generation
│   ├── proximity.py        # Semantic deduplication
│   ├── reflection.py       # Automated peer review
│   ├── ranking.py          # Elo tournament ranking
│   ├── evolution.py        # Hypothesis improvement
│   ├── meta_review.py      # Pattern analysis & bias detection
│   └── supervisor.py       # Central orchestrator
│
├── utils/                  # Utility modules
│   ├── __init__.py
│   ├── llm_client.py       # LLM interface with retry logic
│   ├── literature_search.py # Multi-source academic search
│   └── text_processing.py  # Text embeddings & clustering
│
└── tests/                  # Comprehensive test suite
    ├── __init__.py
    ├── unit/               # Unit tests for each component
    │   ├── test_generation.py
    │   ├── test_proximity.py
    │   ├── test_reflection.py
    │   ├── test_ranking.py
    │   ├── test_evolution.py
    │   ├── test_meta_review.py
    │   └── test_supervisor.py
    └── integration/        # End-to-end integration tests
        └── test_full_system.py
```

## 🔧 Advanced Configuration

### Custom Agent Parameters

```python
# Proximity Agent - Clustering sensitivity
proximity_agent = ProximityAgent(
    distance_threshold=0.15,  # Stricter similarity threshold
    min_cluster_size=2        # Minimum cluster size
)

# Reflection Agent - Review criteria weights
reflection_agent = RobustReflectionAgent(
    criteria_weights={
        'novelty': 0.25,
        'feasibility': 0.20,
        'rigor': 0.20,
        'impact': 0.25,
        'testability': 0.10
    }
)

# Ranking Agent - Tournament parameters
ranking_agent = RankingAgent(
    k_factor=32,              # Elo K-factor
    initial_rating=1200,      # Starting Elo rating
    max_debates=10            # Maximum pairwise comparisons
)

# Evolution Agent - Strategy selection
evolution_agent = EvolutionAgent(
    strategies=['simplify', 'combine', 'analogize'],
    max_evolutions=3
)
```

### Factory Patterns

```python
from agents.supervisor import SupervisorFactory

# Quick prototyping (2-3 cycles)
supervisor = SupervisorFactory.create_rapid_prototyping_supervisor(
    research_goal="Quick validation of AI approach",
    constraints=["Keep simple", "Focus on feasibility"]
)

# Thorough research (5+ cycles with meta-analysis)  
supervisor = SupervisorFactory.create_thorough_research_supervisor(
    research_goal="Comprehensive analysis of ML techniques",
    constraints=["Consider ethical implications", "Validate thoroughly"]
)

# Quality-focused (frequent evolution)
supervisor = SupervisorFactory.create_quality_focused_supervisor(
    research_goal="High-quality hypothesis development",
    constraints=["Prioritize rigor", "Ensure reproducibility"]
)
```

## 📊 Output Formats

### Research Results

The system produces comprehensive research outputs:

```json
{
  "config": {
    "research_goal": "AI applications in healthcare",
    "max_cycles": 3,
    "constraints": ["Focus on patient safety"]
  },
  "final_hypotheses": [
    {
      "id": "evolved_hyp_1",
      "content": "Machine learning models can predict...",
      "rank": 1,
      "elo_rating": 1285.4,
      "confidence": 0.87,
      "evolution_history": ["original_hyp_1", "simplified_hyp_1"]
    }
  ],
  "cycle_history": [
    {
      "cycle": 1,
      "num_proposals": 15,
      "num_unique": 12,
      "best_score": 8.4,
      "duration_sec": 45.2
    }
  ],
  "meta_analysis": {
    "patterns_detected": 4,
    "evaluation_bias": "slight preference for feasibility",
    "recommendations": "Increase diversity in literature sources"
  }
}
```

### Export Options

```python
# Save results to file
supervisor.save_results(results, "research_output.json")

# Export to different formats
supervisor.export_to_csv(results, "hypotheses.csv")
supervisor.export_to_markdown(results, "research_report.md")
```

## 🛠️ Troubleshooting

### Common Issues

1. **API Key Issues**
   ```bash
   # Verify your environment variables
   echo $GOOGLE_API_KEY
   echo $PUBMED_EMAIL
   
   # Test configuration
   python -c "from core.config import get_config; print(get_config())"
   ```

2. **Import Errors**
   ```bash
   # Ensure you're in the project directory
   cd ai_coscientist
   
   # Install missing dependencies
   pip install -r requirements.txt
   ```

3. **Literature Search Failures**
   ```bash
   # Test individual literature sources
   python -c "from utils.literature_search import search_pubmed; print(search_pubmed('AI healthcare'))"
   ```

4. **LLM Connection Issues**
   ```bash
   # Test LLM connection
   python -c "from utils.llm_client import create_llm_client; client = create_llm_client(); print(client.invoke('Test'))"
   ```

### Performance Optimization

- **Reduce API calls**: Use smaller `max_cycles` for testing
- **Enable caching**: Set `CACHE_ENABLED=true` in environment
- **Parallel processing**: Use `--parallel` flag for literature search
- **Mock mode**: Set `USE_MOCK_LLM=true` for development

### Debugging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug mode
config = SupervisorConfig(debug_mode=True)
supervisor = IntegratedSupervisor(config)
```

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```
4. Run tests before committing:
   ```bash
   python run_tests.py
   ```

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Add docstrings for all public functions
- Write tests for new functionality

### Adding New Agents

1. Create agent class inheriting from base agent interface
2. Implement required methods (`run`, `validate_input`, `process_output`)
3. Add comprehensive unit tests
4. Update integration tests
5. Document in README

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 References

- **Multi-Agent Systems**: Wooldridge, M. (2009). An Introduction to MultiAgent Systems
- **Scientific Research Automation**: King, R. D. et al. (2009). The automation of science. Science, 324(5923), 85-89
- **LLM-based Research**: Wei, J. et al. (2022). Chain-of-thought prompting elicits reasoning in large language models
- **Elo Rating Systems**: Elo, A. E. (1978). The rating of chessplayers, past and present

## 📞 Support

For questions, issues, or contributions:

- **Documentation**: See CLAUDE.md for detailed development guidance
- **Issues**: Create an issue on GitHub
- **Email**: Contact the development team

---

**Built with ❤️ for advancing autonomous scientific research**