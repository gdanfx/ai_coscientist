# AI Co-Scientist Notebook Migration Plan - Optimal Strategy

This YAML-based migration plan for converting the 10-cell Jupyter notebook into a modular Python project.

**Execution Strategy:** Dependency-first, risk-minimized, test-integrated approach

## Project Information

```yaml
project_info:
  name: "AI Co-Scientist Migration"
  source: "10-cell Jupyter notebook implementing multi-agent AI research system"
  target: "Modular Python project with clear separation of concerns"
  risk_level: "Medium"
```

Here’s project file structure.  lean, modular, yet minimal layout. It keeps only what we need to split the notebook cleanly. nothing extra, nothing over-engineered.

ai_coscientist/                 # ← project root (importable as `ai_coscientist`)
├── .env.example                # template for API keys / secrets  (never commit real .env)
├── .gitignore                  # standard Python / OS ignores
├── README.md                   # setup + run instructions
├── requirements.txt            # pinned dependencies
├── main.py                     # one-liner CLI/entry point (calls Supervisor)
│
├── core/                       # shared, non-agent-specific plumbing
│   ├── __init__.py
│   ├── config.py               # loads .env, holds constants
│   └── data_structures.py      # all @dataclass state objects (GenerationState, etc.)
│
├── agents/                     # one file per agent; keep logic self-contained
│   ├── __init__.py
│   ├── generation.py           # literature search + hypothesis generation
│   ├── proximity.py            # clustering & deduplication
│   ├── reflection.py           # peer-review evaluation
│   ├── ranking.py              # Elo-style tournament ranking
│   ├── evolution.py            # hypothesis improvement strategies
│   ├── meta_review.py          # system-level analysis
│   └── supervisor.py           # orchestrates full research cycles
│
├── utils/                      # lightweight helpers (only if truly shared)
│   ├── __init__.py
│   ├── literature_search.py    # PubMed, arXiv, CrossRef wrappers
│   ├── llm_client.py           # single LLM interface / retry logic
│   └── text_processing.py      # embeddings, clustering helpers
│
├── tests/                      # PyTest suite mirrors structure
│   ├── __init__.py
│   ├── unit/                   # fast, isolated agent tests
│   │   ├── test_generation.py
│   │   ├── test_proximity.py
│   │   └── …                  # one per agent / util
│   └── integration/            # end-to-end system tests (mocks for APIs)
│       └── test_full_system.py
│
└── notebooks/                  # original Jupyter notebooks for reference only
    └── ai_coscientist_full.ipynb


## Migration Plan

### PHASE 1: FOUNDATION & INFRASTRUCTURE

**Critical foundation files that everything else depends on**

#### requirements.txt
- [ ] **File Creation** | **Complexity:** 1 | **Priority:** 10 | **Testing Effort:** 0 | **Dependencies:** 0
- **Phase:** 1 | **Estimated Hours:** 1
- **Source:** Cell 1: Setup & Configuration
- **Purpose:** Pin all project dependencies for reproducible environments
- **Migration Notes:**
  - [ ] Extract from notebook pip install commands
  - [ ] Pin specific versions for reproducibility
  - [ ] Include: langchain-community, duckduckgo-search, langgraph, sentence-transformers
- **Success Criteria:**
  - [ ] All dependencies install without conflicts
  - [ ] Virtual environment setup works cleanly

#### .env.example
- [ ] **File Creation** | **Complexity:** 1 | **Priority:** 9 | **Testing Effort:** 0 | **Dependencies:** 0
- **Phase:** 1 | **Estimated Hours:** 0.5
- **Source:** Cell 1: Setup & Configuration
- **Purpose:** Template for environment variables and API keys
- **Migration Notes:**
  - [ ] GOOGLE_API_KEY=your_gemini_api_key_here
  - [ ] PUBMED_EMAIL=your_email@example.com
  - [ ] Never commit real .env file
- **Success Criteria:**
  - [ ] Template covers all required environment variables
  - [ ] Clear documentation of where to obtain API keys

#### .gitignore
- [ ] **File Creation** | **Complexity:** 1 | **Priority:** 8 | **Testing Effort:** 0 | **Dependencies:** 0
- **Phase:** 1 | **Estimated Hours:** 0.5
- **Source:** Standard Python gitignore
- **Purpose:** Prevent sensitive files from being committed
- **Migration Notes:**
  - [ ] Include .env, __pycache__, .pytest_cache
  - [ ] Exclude notebook checkpoints and model files
- **Success Criteria:**
  - [ ] No sensitive files can be accidentally committed

#### core/config.py
- [ ] **File Creation** | **Complexity:** 2 | **Priority:** 9 | **Testing Effort:** 2 | **Dependencies:** 0
- **Phase:** 1 | **Estimated Hours:** 2
- **Source:** Cell 1: Setup & Configuration
- **Purpose:** Centralize configuration management and environment loading
- **Migration Notes:**
  - [ ] Load environment variables with python-dotenv
  - [ ] Provide sensible defaults and validation
  - [ ] Export RATE_LIMIT_DELAY, MAX_RESULTS_PER_SOURCE constants
- **Success Criteria:**
  - [ ] Environment variables load correctly
  - [ ] Configuration validation works
  - [ ] Module can be imported without external dependencies

#### core/data_structures.py
- [ ] **File Creation** | **Complexity:** 3 | **Priority:** 10 | **Testing Effort:** 3 | **Dependencies:** 0
- **Phase:** 1 | **Estimated Hours:** 4
- **Source:** 
  - Cell 1: GenerationState
  - Cell 4: ReviewCriteria, HypothesisReview, ReflectionState
  - Cell 5: PairwiseComparison, RankingState
  - Cell 6: EvolutionStep, EvolvedHypothesis, EvolutionState
  - Cell 7: MetaReviewState
  - Cell 8: ProximityState
  - Cell 9: SupervisorConfig, SupervisorState
- **Purpose:** Define all shared data structures for inter-agent communication
- **Migration Notes:**
  - [ ] Use dataclasses for all state objects
  - [ ] Include type hints and validation
  - [ ] Keep pure data - no methods or business logic
- **Success Criteria:**
  - [ ] All dataclasses validate correctly
  - [ ] No circular imports
  - [ ] Complete coverage of notebook state objects

### PHASE 2: UTILITIES & SHARED COMPONENTS

**Reusable components that agents depend on**

#### utils/llm_client.py
- [ ] **File Creation** | **Complexity:** 4 | **Priority:** 9 | **Testing Effort:** 6 | **Dependencies:** 1
- **Phase:** 2 | **Estimated Hours:** 6
- **Dependencies:** core/config.py
- **Source:** Cell 1: LLM setup, Various cells: LLM interactions
- **Purpose:** Unified LLM interface with retry logic and error handling
- **Migration Notes:**
  - [ ] Centralize ChatGoogleGenerativeAI initialization
  - [ ] Implement exponential backoff for rate limiting
  - [ ] Add consistent error handling and logging
- **Success Criteria:**
  - [ ] LLM client initializes successfully
  - [ ] Retry logic handles API failures gracefully
  - [ ] Easy to mock for testing

#### utils/literature_search.py
- [ ] **File Creation** | **Complexity:** 6 | **Priority:** 8 | **Testing Effort:** 8 | **Dependencies:** 1
- **Phase:** 2 | **Estimated Hours:** 8
- **Dependencies:** core/config.py
- **Source:** Cell 2: Multi-source literature search functions
- **Purpose:** Unified interface for external academic database APIs
- **Migration Notes:**
  - [ ] Extract search_pubmed, search_arxiv, search_crossref, search_web_academic
  - [ ] Implement robust error handling for each API
  - [ ] Add rate limiting and request timeout handling
- **Success Criteria:**
  - [ ] All search functions work independently
  - [ ] Graceful failure when APIs are unavailable
  - [ ] Easy to mock for testing

#### utils/text_processing.py
- [ ] **File Creation** | **Complexity:** 5 | **Priority:** 7 | **Testing Effort:** 5 | **Dependencies:** 0
- **Phase:** 2 | **Estimated Hours:** 5
- **Source:** 
  - Cell 8: SentenceTransformer embeddings
  - Cell 7: TF-IDF and clustering
- **Purpose:** Text processing utilities for embeddings and clustering
- **Migration Notes:**
  - [ ] Lazy load SentenceTransformer models
  - [ ] Implement caching for embeddings
  - [ ] Extract clustering algorithms
- **Success Criteria:**
  - [ ] Embeddings generate consistently
  - [ ] Clustering produces stable results
  - [ ] Models load efficiently

### PHASE 3: CORE AGENTS IMPLEMENTATION

**Implement agents in dependency order, with tests for each**

#### agents/generation.py
- [ ] **File Creation** | **Complexity:** 7 | **Priority:** 9 | **Testing Effort:** 8 | **Dependencies:** 4
- **Phase:** 3 | **Estimated Hours:** 12
- **Dependencies:** core/data_structures.py, utils/llm_client.py, utils/literature_search.py
- **Source:** Cell 2: Enhanced Multi-Source Literature Search Agent
- **Purpose:** Literature search orchestration and hypothesis generation
- **Migration Notes:**
  - [ ] Extract EnhancedGenerationAgent class
  - [ ] Implement run_complete_workflow method
  - [ ] Handle multi-source search coordination
- **Success Criteria:**
  - [ ] Generates valid GenerationState objects
  - [ ] Handles API failures gracefully
  - [ ] Produces diverse initial hypotheses
- **Test File:** tests/unit/test_generation.py
- **Test Source:** Cell 3: Literature Search Interface Validation Test

#### agents/proximity.py
- [ ] **File Creation** | **Complexity:** 6 | **Priority:** 7 | **Testing Effort:** 6 | **Dependencies:** 2
- **Phase:** 3 | **Estimated Hours:** 8
- **Dependencies:** core/data_structures.py, utils/text_processing.py
- **Source:** Cell 8: Proximity Agent - Clustering & Deduplication
- **Purpose:** Semantic deduplication and diversity maintenance
- **Migration Notes:**
  - [ ] Extract ProximityAgent class
  - [ ] Implement embedding and clustering logic
  - [ ] Handle edge cases with small datasets
- **Success Criteria:**
  - [ ] Correctly identifies similar hypotheses
  - [ ] Maintains intellectual diversity
  - [ ] Handles edge cases gracefully
- **Test File:** tests/unit/test_proximity.py

#### agents/reflection.py
- [ ] **File Creation** | **Complexity:** 8 | **Priority:** 8 | **Testing Effort:** 8 | **Dependencies:** 2
- **Phase:** 3 | **Estimated Hours:** 10
- **Dependencies:** core/data_structures.py, utils/llm_client.py
- **Source:** Cell 4: Robust Reflection Agent
- **Purpose:** Automated peer review and hypothesis evaluation
- **Migration Notes:**
  - [ ] Extract robust parsing logic
  - [ ] Implement fallback mechanisms
  - [ ] Handle varied LLM output formats
- **Success Criteria:**
  - [ ] Parses LLM outputs robustly
  - [ ] Provides consistent review scores
  - [ ] Fallback mechanisms work correctly
- **Test File:** tests/unit/test_reflection.py

#### agents/ranking.py
- [ ] **File Creation** | **Complexity:** 8 | **Priority:** 7 | **Testing Effort:** 7 | **Dependencies:** 2
- **Phase:** 3 | **Estimated Hours:** 10
- **Dependencies:** core/data_structures.py, utils/llm_client.py
- **Source:** Cell 5: Ranking Agent - Elo Tournament System
- **Purpose:** Competitive ranking through pairwise debates
- **Migration Notes:**
  - [ ] Extract EloSystem mathematics
  - [ ] Implement tournament logic
  - [ ] Handle debate parsing edge cases
- **Success Criteria:**
  - [ ] Elo ratings update correctly
  - [ ] Tournament produces stable rankings
  - [ ] Debate parsing is robust
- **Test File:** tests/unit/test_ranking.py

#### agents/evolution.py
- [ ] **File Creation** | **Complexity:** 8 | **Priority:** 7 | **Testing Effort:** 7 | **Dependencies:** 2
- **Phase:** 3 | **Estimated Hours:** 12
- **Dependencies:** core/data_structures.py, utils/llm_client.py
- **Source:** Cell 6: Evolution Agent - Multi-Strategy Hypothesis Improvement
- **Purpose:** Creative hypothesis improvement through evolution strategies
- **Migration Notes:**
  - [ ] Extract all evolution strategies
  - [ ] Implement strategy selection logic
  - [ ] Track evolution effectiveness
- **Success Criteria:**
  - [ ] All strategies produce valid outputs
  - [ ] Strategy effectiveness tracking works
  - [ ] Evolution improves hypothesis quality
- **Test File:** tests/unit/test_evolution.py

#### agents/meta_review.py
- [ ] **File Creation** | **Complexity:** 7 | **Priority:** 6 | **Testing Effort:** 6 | **Dependencies:** 2
- **Phase:** 3 | **Estimated Hours:** 8
- **Dependencies:** core/data_structures.py, utils/text_processing.py
- **Source:** Cell 7: Enhanced Meta-Review Agent (Fixed)
- **Purpose:** System-level analysis and bias detection
- **Migration Notes:**
  - [ ] Extract correlation analysis logic
  - [ ] Implement pattern detection
  - [ ] Handle statistical edge cases
- **Success Criteria:**
  - [ ] Pattern detection works correctly
  - [ ] Statistical analysis is robust
  - [ ] Provides actionable insights
- **Test File:** tests/unit/test_meta_review.py

### PHASE 4: SYSTEM INTEGRATION

**Build supervisor and orchestrate complete system**

#### agents/supervisor.py
- [ ] **File Creation** | **Complexity:** 9 | **Priority:** 9 | **Testing Effort:** 9 | **Dependencies:** 8
- **Phase:** 4 | **Estimated Hours:** 16
- **Dependencies:** core/config.py, core/data_structures.py, All agent modules
- **Source:** 
  - Cell 9: Supervisor Agent (Stubbed Demo)
  - Cell 10: Full System Integration
- **Purpose:** Central orchestrator for complete research cycles
- **Migration Notes:**
  - [ ] Extract supervisor control loop
  - [ ] Implement early stopping logic
  - [ ] Handle agent failures gracefully
- **Success Criteria:**
  - [ ] Orchestrates all agents correctly
  - [ ] Early stopping works as expected
  - [ ] Error recovery mechanisms function
- **Test File:** tests/unit/test_supervisor.py

#### main.py
- [ ] **File Creation** | **Complexity:** 2 | **Priority:** 10 | **Testing Effort:** 2 | **Dependencies:** 2
- **Phase:** 4 | **Estimated Hours:** 3
- **Dependencies:** agents/supervisor.py, core/config.py
- **Source:** Cell 10: Main execution logic
- **Purpose:** Command-line entry point for the system
- **Migration Notes:**
  - [ ] Simple CLI interface
  - [ ] Configuration loading
  - [ ] Basic argument parsing
- **Success Criteria:**
  - [ ] System runs end-to-end from CLI
  - [ ] Configuration loads correctly
  - [ ] Error messages are helpful

### PHASE 5: TESTING & VALIDATION

**Complete testing suite and final validation**

#### tests/integration/test_full_system.py
- [ ] **File Creation** | **Complexity:** 9 | **Priority:** 8 | **Testing Effort:** 10 | **Dependencies:** 8
- **Phase:** 5 | **Estimated Hours:** 12
- **Dependencies:** All system modules
- **Source:** Cell 10: Full System Integration test
- **Purpose:** End-to-end system validation
- **Migration Notes:**
  - [ ] Mock all external APIs
  - [ ] Test complete research cycles
  - [ ] Validate error handling
- **Success Criteria:**
  - [ ] Full system runs without errors
  - [ ] All integration points work
  - [ ] Error scenarios handled correctly

#### README.md
- [ ] **File Creation** | **Complexity:** 3 | **Priority:** 6 | **Testing Effort:** 0 | **Dependencies:** 0
- **Phase:** 5 | **Estimated Hours:** 4
- **Source:** Documentation from notebook
- **Purpose:** Comprehensive project documentation
- **Migration Notes:**
  - [ ] Setup and installation instructions
  - [ ] API key configuration guide
  - [ ] Usage examples and architecture overview
- **Success Criteria:**
  - [ ] New users can set up project from README
  - [ ] All features documented
  - [ ] Troubleshooting guide included

## Implementation Guidelines

### Approach & Philosophy
- **Approach:** Incremental development with continuous testing
- **Testing Philosophy:** Test immediately after each component
- **Error Handling:** Graceful degradation with informative messages

### Critical Success Factors
- [ ] Complete Phase 1 before starting Phase 2
- [ ] Test each agent immediately after implementation
- [ ] Mock all external APIs for reliable testing
- [ ] Maintain clear separation of concerns
- [ ] Document any deviations from original notebook behavior

### Risk Mitigation

#### High Risk Areas
- [ ] LLM API integration and rate limiting
- [ ] Multi-source literature search coordination
- [ ] Complex agent interaction in supervisor

#### Mitigation Strategies
- [ ] Implement comprehensive mocking
- [ ] Add circuit breaker patterns for external APIs
- [ ] Create detailed integration tests
- [ ] Build fail-safe mechanisms in supervisor

### Quality Gates
- [ ] **Phase 1:** All core modules import and basic config works
- [ ] **Phase 2:** All utilities work independently with mocked externals
- [ ] **Phase 3:** Each agent passes unit tests and integrates correctly
- [ ] **Phase 4:** Full system runs at least one complete cycle
- [ ] **Phase 5:** All tests pass and documentation is complete

## Success Metrics

### Code Quality
- [ ] All modules have >90% test coverage
- [ ] No circular dependencies
- [ ] Consistent error handling patterns

### Functionality
- [ ] System produces research outputs equivalent to notebook
- [ ] All original notebook capabilities preserved
- [ ] Performance within 20% of original notebook

### Maintainability
- [ ] Clear module boundaries and interfaces
- [ ] Comprehensive documentation
- [ ] Easy to extend with new agents or strategies

