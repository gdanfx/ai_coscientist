This Jupyter notebook implements a sophisticated multi-agent AI system designed to function as an "AI Co-Scientist" that automates the complete scientific research cycle. The system orchestrates multiple specialized agents working together to generate, evaluate, rank, and evolve scientific hypotheses through iterative cycles.

**Core Philosophy**: The system mimics the human scientific process: literature review → hypothesis generation → peer review → competitive ranking → creative improvement → meta-analysis, creating a self-improving AI scientist capable of conducting research autonomously.

## **System Architecture & Agent Flow**

\+---------------------+      \+------------------+      \+------------------+  
| Generation Agent    | \---\> | Proximity Agent  | \---\> | Reflection Agent |  
\+---------------------+      \+------------------+      \+------------------+  
        ^                                                    |  
        |                                                    v  
\+---------------------+      \+------------------+      \+------------------+  
| Evolution Agent     | \<--- | Ranking Agent    | \<--- | Meta-Review      |  
\+---------------------+      \+------------------+      \+------------------+  
                ^                                   (Supervisor orchestrates)  
                '----------------------------------------------'

**System-Wide Logic Flow**:

1. **Input**: Research goal \+ constraints  
2. **Literature Search**: Gather relevant papers from multiple sources  
3. **Initial Generation**: Create hypotheses based on literature synthesis  
4. **Deduplication**: Remove similar ideas (Proximity Agent)  
5. **Evaluation**: Score hypotheses on multiple criteria (Reflection Agent)  
6. **Competition**: Rank through pairwise debates (Ranking Agent)  
7. **Evolution**: Improve top hypotheses using various strategies  
8. **Meta-Analysis**: Identify system patterns and suggest improvements  
9. **Iteration**: Repeat cycle with evolved hypotheses  
10. **Convergence**: Stop when no further improvement or max cycles reached

---

## **Cell-by-Cell Technical Breakdown**

### **Cell 1: Setup & Configuration `[2GDBt3erMBG4]`**

**Purpose**: Foundational infrastructure setup for the entire multi-agent system.

**Key Components**:

* **Dependencies**: Installs LangChain, DuckDuckGo search, LangGraph, Google Generative AI  
* **Configuration Variables**:  
  * `PUBMED_EMAIL`: Required for PubMed API access  
  * `RATE_LIMIT_DELAY`: Controls API call frequency to avoid blocking  
  * `MAX_RESULTS_PER_SOURCE`: Limits results per literature source  
* **Core Data Structure**: `GenerationState` dataclass acts as shared memory between agents  
* **LLM Setup**: Initializes Gemini LLM model (`gemini-2.0-flash-exp`) with API key management

**Code Structure**:

\# Package installation  
\!pip install langchain-community duckduckgo-search langgraph langchain-google-genai

\# Essential imports  
from langchain\_community.tools import DuckDuckGoSearchRun  
from langgraph.graph import StateGraph  
from langchain\_google\_genai import ChatGoogleGenerativeAI

\# Configuration constants  
PUBMED\_EMAIL \= "your\_email@example.com"  
RATE\_LIMIT\_DELAY \= 1.0  
MAX\_RESULTS\_PER\_SOURCE \= 3

\# Shared state structure  
@dataclass  
class GenerationState:  
    research\_goal: str  
    constraints: List\[str\]  
    search\_queries: List\[str\]  
    literature\_findings: List\[Dict\]  
    synthesized\_knowledge: str  
    generated\_proposals: List\[Dict\]

**Outputs**:

* Configured environment with all necessary libraries  
* Initialized Gemini LLM model with error handling  
* Setup confirmation messages  
* Warning about dependency resolver (common, non-critical)

**High-Level Logic**: Establishes the infrastructure needed for multi-source literature search and LLM-based hypothesis generation, ensuring all downstream agents have consistent dependencies and shared constants.

---

### **Cell 2: Enhanced Multi-Source Literature Search Agent `[L56ykRxEMCWZ]`**

**Purpose**: Implements the first core component that gathers relevant scientific literature and generates initial hypotheses.

**Core Functions & Classes**:

1. **Search Functions** (Multi-source data collection):

   * `search_pubmed(query, max_results=3)`: NCBI eUtils API integration  
     * Input: Search query string, result limit  
     * Output: List of dictionaries containing paper metadata  
     * Logic: Uses NCBI eUtils API → fetches PMIDs → retrieves detailed article info → parses XML → extracts title, abstract, authors, journal, year, key findings  
   * `search_arxiv(query, max_results=3)`: ArXiv preprint database  
     * Input: Search query, result limit  
     * Output: List of preprint papers from ArXiv  
     * Logic: Queries ArXiv API → parses XML response → extracts paper details  
   * `search_crossref(query, max_results=3)`: Academic papers database  
     * Input: Search query, result limit  
     * Output: Academic papers from CrossRef database  
     * Logic: Uses CrossRef REST API → finds published papers → standardizes format  
   * `search_web_academic(query, max_results=3)`: Web search with academic focus  
     * Logic: Uses DuckDuckGo with `site:` operators to restrict searches to academic domains  
2. **Multi-Source Orchestration**:

   * `multi_source_literature_search(queries)`:  
     * Input: List of search queries  
     * Output: Deduplicated list of papers from all sources  
     * Logic: Runs searches across all sources → removes duplicates based on title similarity → sorts by relevance and citation count → implements intelligent fallbacks when sources fail

**Enhanced Generation Agent Class**:

 class EnhancedGenerationAgent:  
    def run\_complete\_workflow(self, research\_goal, constraints):  
        \# Generate optimized search queries  
        \# Search multiple literature sources    
        \# Synthesize knowledge from findings  
        \# Generate novel hypotheses based on synthesis

3. 

**Interface Contract**: The comprehensive docstring defines exact input/output specifications:

* **Input**: `research_goal` (str), `constraints` (List\[str\])  
* **Output**: `GenerationState` with populated fields for search queries, literature findings, synthesized knowledge, and generated proposals

**Error Handling**: Graceful degradation when APIs fail, rate limiting awareness, duplicate detection and removal.

**Outputs**:

* `🔍 Generating multi-source search queries...`: Mock LLM provides 4 optimized queries  
* `📚 Searching multiple literature sources...`: Agent queries different APIs  
* `WARNING: No meaningful web results...`: Demonstrates real-world API behavior handling  
* `✅ Found 12 papers from 1 sources:`: Successful search results (realistic that only some sources return results)  
* `💡 Generated Hypotheses:`: Final novel hypothesis based on literature synthesis

**High-Level Logic**: Acts as the "input engine" to the system, gathering relevant scientific literature and generating initial hypotheses based on current research. Hides all API complexities behind a clean interface for the supervisor.

---

### **Cell 3: Literature Search Interface Validation Test `[Af-8uN7xEfL4]`**

**Purpose**: Formal unit test ensuring the Literature Search Agent adheres to its declared interface contract.

**Code Structure**:

def test\_literature\_search\_interface():  
    \# Try real agent first, fall back to mock if needed  
    try:  
        agent \= EnhancedGenerationAgent()  
    except NameError:  
        agent \= MockEnhancedGenerationAgent()  
      
    \# Execute with standard test inputs  
    result \= agent.run\_complete\_workflow(test\_research\_goal, test\_constraints)  
      
    \# Validate output structure programmatically  
    assert hasattr(result, 'search\_queries')  
    assert hasattr(result, 'literature\_findings')  
    \# ... more validations

**Validation Checks**:

* Presence of all required fields in `GenerationState`  
* Correct data types for each field  
* Proper structure of nested objects (literature findings with 'title', 'source', 'url')  
* Content quality validation

**Outputs**:

* `✅ Using real EnhancedGenerationAgent`: Confirms using actual implementation  
* `WARNING...Rate limited...`: Demonstrates real-world API behavior (DuckDuckGo rate limiting)  
* **Validation Steps**: Logs each validation check with pass/fail status  
* `✅ INTERFACE CONTRACT VALIDATION: PASSED`: Critical confirmation for integration readiness

**High-Level Logic**: Catches integration errors early, ensuring reliable integration into the supervisor-controlled system before complex multi-agent interactions begin.

---

### **Cell 4: Robust Reflection Agent `[lNtZrhutOgBA]`**

**Purpose**: Acts as an automated peer reviewer, critically evaluating hypotheses using multiple scientific criteria.

**Core Data Structures**:

@dataclass  
class ReviewCriteria:  
    novelty: float         \# 1-10 scale  
    feasibility: float     \# 1-10 scale    
    scientific\_rigor: float \# 1-10 scale  
    impact: float          \# 1-10 scale  
    testability: float     \# 1-10 scale

@dataclass    
class HypothesisReview:  
    hypothesis\_id: str  
    criteria: ReviewCriteria  
    strengths: List\[str\]  
    weaknesses: List\[str\]  
    recommendations: List\[str\]  
    overall\_assessment: str  
    confidence: float

@dataclass  
class ReflectionState:  
    reviews: List\[HypothesisReview\]  
    statistics: Dict  
    summary: str

**Key Methods**:

1. **`_parse_flexible_review`**: Core robustness mechanism

   * Uses multiple Regular Expression patterns to extract scores and sections  
   * Handles variations in LLM output format  
   * Provides fallback parsing strategies  
2. **`_create_enhanced_fallback_review`**: Smart error handling

   * When parsing fails completely, generates heuristic review  
   * Uses keyword analysis of hypothesis text  
   * Prevents system crashes, allows continued operation  
3. **`quick_review(hypothesis_text, hypothesis_id)`**: Rapid evaluation

   * Input: Hypothesis content and ID  
   * Output: HypothesisReview object  
   * Logic: Simplified prompts for fast assessment  
4. **`detailed_review(hypothesis_text, hypothesis_id, research_goal)`**: Comprehensive evaluation

   * Input: Hypothesis details and research context  
   * Output: Comprehensive HypothesisReview  
   * Logic: Evaluates across 5 criteria → identifies strengths/weaknesses → provides detailed reasoning  
5. **`adaptive_batch_review(hypotheses, research_goal)`**: Main orchestration

   * Input: List of hypotheses, research context  
   * Output: ReflectionState with all reviews  
   * Logic: Determines review type based on initial scores → computes statistics → identifies quality patterns

**Outputs**:

* `ROBUST BATCH SUMMARY`: High-level overview showing quality distribution  
* `SAMPLE ROBUST REVIEW`: Detailed breakdown for individual hypotheses  
* Structured scores with unstructured assessments  
* Critical analysis capability (can assign very low scores like 1.00/10)

**High-Level Logic**: Functions as a systematic "peer reviewer" that evaluates hypothesis quality across multiple dimensions, providing the foundation for ranking and evolution decisions. The "robust" design ensures it can handle varied LLM outputs reliably.

---

### **Cell 5: Ranking Agent \- Elo Tournament System `[KBq0rM8-Rszy]`**

**Purpose**: Creates definitive hypothesis rankings through competitive pairwise comparisons using Elo rating system.

**Core Components**:

**`EloSystem` Class**: Implements chess-like rating mathematics

 class EloSystem:  
    def update\_ratings(self, rating\_a, rating\_b, actual\_score\_a):  
        \# Calculate expected scores  
        expected\_a \= 1.0 / (1.0 \+ 10\*\*((rating\_b \- rating\_a) / 400))  
        \# Update ratings based on actual vs expected performance  
        rating\_a \+= K \* (actual\_score\_a \- expected\_a)

1.   
2. **`RankingAgent` Class**:

   * **`pairwise_debate(hypothesis_a, hypothesis_b, round_number)`**:

     * Input: Two hypothesis reviews and round info  
     * Output: PairwiseComparison with winner and reasoning  
     * Logic: Constructs debate prompt → LLM conducts structured debate (advocate for A, advocate for B, critical analysis) → determines winner and confidence → updates Elo ratings  
   * **`run_full_tournament(reflection_state, num_rounds=3)`**:

     * Input: Hypothesis reviews, tournament parameters  
     * Output: RankingState with final rankings  
     * Logic: Initialize Elo ratings (adjusted by reflection scores) → run multiple rounds of pairwise comparisons → generate final rankings sorted by Elo rating → compute tournament statistics  
3. **`_parse_debate_result`**: Robust result parsing

   * Uses regex to extract winner, confidence, reasoning  
   * Includes fallback logic using reflection scores when LLM choice unclear

**Analysis Functions**:

* `_generate_final_rankings`: Creates sorted leaderboard  
* `_compute_ranking_statistics`: Tournament convergence metrics  
* Decision quality analysis and recommendations

**Outputs**:

* Detailed tournament log with round-by-round results  
* Elo rating changes over time  
* `Final Rankings`: Leaderboard with Elo scores and win/loss records  
* `TOURNAMENT SUMMARY`: High-level report with process recommendations

**High-Level Logic**: Creates a competitive ranking system where hypotheses "compete" against each other through structured debates, producing a reliable quality hierarchy that goes beyond simple score sorting.

---

### **Cell 6: Evolution Agent \- Multi-Strategy Hypothesis Improvement `[1Dyp7myfTwz5]`**

**Purpose**: Creative engine that improves top-ranked hypotheses through various evolutionary strategies.

**Evolution Strategies**:

class EvolutionStrategy(Enum):  
    SIMPLIFY \= "simplify"  
    COMBINE \= "combine"   
    ANALOGICAL \= "analogical"  
    RADICAL\_VARIATION \= "radical\_variation"  
    CONSTRAINT\_RELAXATION \= "constraint\_relaxation"

**Core Strategy Functions**:

1. **`simplify_hypothesis(hypothesis_review, ranking_info)`**:

   * Input: Hypothesis review and ranking position  
   * Output: EvolutionStep with simplified version  
   * Logic: Reduces complexity while maintaining core innovation  
2. **`combine_hypotheses(hypothesis_reviews, ranking_infos)`**:

   * Input: Multiple top hypotheses  
   * Output: EvolutionStep with hybrid approach  
   * Logic: Integrates best elements from different hypotheses  
3. **`analogical_reasoning(hypothesis_review, ranking_info)`**:

   * Input: Single hypothesis and context  
   * Output: EvolutionStep inspired by other domains  
   * Logic: Applies successful patterns from other scientific fields  
4. **`radical_variation(hypothesis_review, ranking_info)`**:

   * Input: Hypothesis to transform  
   * Output: EvolutionStep with paradigm-shifting approach  
   * Logic: Creates fundamentally different approach to same problem (explicitly challenges conventional assumptions)  
5. **`constraint_relaxation(hypothesis_review, ranking_info, constraints)`**:

   * Input: Hypothesis and original constraints  
   * Output: EvolutionStep with expanded possibilities  
   * Logic: Relaxes key constraints to unlock new approaches

**Data Structures**:

@dataclass  
class EvolutionStep:  
    strategy: EvolutionStrategy  
    parent\_ids: List\[str\]  
    evolved\_content: str  
    rationale: str  
    predicted\_improvements: Dict

@dataclass    
class EvolvedHypothesis:  
    hypothesis\_id: str  
    content: str  
    evolution\_history: List\[EvolutionStep\]  
      
@dataclass  
class EvolutionState:  
    evolved\_hypotheses: List\[EvolvedHypothesis\]  
    strategy\_effectiveness: Dict  
    evolution\_statistics: Dict

**Main Method**: `evolve_top_hypotheses(ranking_state, reflection_state)`

* Input: Results from ranking and reflection agents  
* Output: EvolutionState with evolved hypotheses  
* Logic: Select top N hypotheses → apply different strategies based on rank and characteristics → track evolution lineage → generate strategy effectiveness statistics

**Outputs**:

* Individual strategy testing results  
* `FULL EVOLUTION PROCESS`: 6 new evolved hypotheses generated  
* `Strategy Effectiveness`: Meta-analysis of average score and novelty gains per strategy  
* `EVOLVED HYPOTHESES`: Display of new creative ideas with their lineage  
* Final summary with recommendations for best new approaches

**High-Level Logic**: Acts as a "creative mutation" system that takes promising ideas and transforms them into potentially better versions using various innovation strategies, similar to genetic algorithms but optimized for scientific concepts.

---

### **Cell 7: Enhanced Meta-Review Agent (Fixed) `[eQYn31TNBvlC]`**

**Purpose**: Analyzes patterns across the entire review process to provide system-level insights and improvements.

**Key Improvements in Fixed Version**:

* More robust TF-IDF parameters for small datasets  
* Better correlation calculation with variance checking  
* Improved error handling for edge cases

**Core Functions**:

1. **`_improved_text_clustering(corpus, n_clusters)`**:

   * Input: List of review texts, desired cluster count  
   * Output: Clustered groups and metrics  
   * Logic: Uses TF-IDF vectorization → applies AgglomerativeClustering with cosine similarity → handles edge cases with fallback methods  
2. **`_compute_correlations_robust(reflection_state, ranking_state)`**:

   * Input: Review and ranking data  
   * Output: Correlation coefficients between criteria and rankings  
   * Logic: **Critical Fix**: Checks for variance before correlation calculation → handles cases where all scores are identical → prevents undefined correlation errors  
3. **`_preprocess_text_robust`**: Enhanced text cleaning before analysis

4. **`run_enhanced_meta_review(reflection_state, ranking_state)`**:

   * Input: Complete review and ranking results  
   * Output: MetaReviewState with patterns and recommendations  
   * Logic: Extracts text corpus from reviews → performs clustering to identify patterns → computes correlations between review scores and rankings → generates actionable feedback

**Analysis Capabilities**:

* Identifies systematic biases in the review process  
* Discovers correlations between review criteria and final rankings  
* Detects patterns in review language and themes  
* Provides recommendations for system improvement

**Outputs**:

* Analysis quality metrics  
* Number of patterns and correlations found  
* Clustering method used (`sklearn_agglomerative_improved`)  
* Sample pattern analysis (e.g., "Cluster 1: Themes \- hypothesis, validation, good, focus, creative")  
* Final report with system-level insights

**High-Level Logic**: Provides "meta-cognition" for the system, identifying how hypotheses are evaluated and suggesting improvements to the review process. Unlike the Reflection Agent which critiques individual hypotheses, this analyzes the entire evaluation system itself.

---

### **Cell 8: Proximity Agent \- Clustering & Deduplication `[s-fRTCXrs-XR]`**

**Purpose**: Maintains intellectual diversity by identifying and removing semantically similar hypotheses.

**Core Technology**:

\# Installs sentence-transformers for advanced text embeddings  
\!pip install sentence-transformers

class ProximityAgent:  
    def \_\_init\_\_(self, distance\_threshold=0.3):  
        \# Loads pre-trained sentence transformer model  
        self.model \= SentenceTransformer('all-MiniLM-L6-v2')  
        self.distance\_threshold \= distance\_threshold

**Key Methods**:

1. **`_embed_hypotheses(proposals)`**:

   * Input: List of hypothesis dictionaries  
   * Output: Numerical embeddings array  
   * Logic: Uses SentenceTransformer to convert text to high-quality semantic vectors  
2. **`_cluster_hypotheses(embeddings)`**:

   * Input: Hypothesis embeddings  
   * Output: Cluster labels  
   * Logic: Uses AgglomerativeClustering with cosine distance metric → groups based on semantic similarity → doesn't require pre-specifying number of clusters  
3. **`_select_representatives`**:

   * For each cluster, selects the hypothesis closest to cluster center  
   * Discards redundant hypotheses while maintaining diversity  
4. **`run(proposals)`**:

   * Input: List of hypothesis proposals  
   * Output: ProximityState with unique hypotheses  
   * Logic: Convert to embeddings → cluster similar hypotheses → select representatives → remove duplicates

**Integration Function**: `run_proximity_agent()` \- Easy plug-in for supervisor loop

**Outputs**:

* Starts with 6 mock hypotheses (with intentionally similar 'a' and 'b' groups)  
* `Cluster Breakdown`: Shows correct grouping of similar ideas  
* `Final Deduplicated Hypotheses`: 3 unique ideas remain (3 redundant ones removed)  
* Demonstrates successful semantic similarity detection

**High-Level Logic**: Ensures the system doesn't waste computational resources on nearly-identical ideas by using advanced NLP techniques to identify and consolidate similar hypotheses while preserving intellectual diversity.

---

### **Cell 9: Supervisor Agent (Stubbed Demo) `[3cA5nounnEvJ]`**

**Purpose**: Central orchestrator that coordinates all agents in structured research cycles. This version uses stubs for testing control logic.

**Core Components**:

**`SupervisorConfig` Class**: Configuration for supervisor behavior

 @dataclass  
class SupervisorConfig:  
    max\_cycles: int \= 3  
    evolution\_every: int \= 2  \# Run evolution every N cycles  
    meta\_every: int \= 3       \# Run meta-review every N cycles  
    no\_improve\_patience: int \= 2  \# Early stopping criteria

1.   
2. **Stub Functions**:

   * **`_ensure_stub`**: Clever helper that checks if real agent functions exist → creates simple placeholders if not → makes cell self-contained for testing  
   * **`_preserve_proposals`**: Bug fix to ensure proposal lists survive stubbed calls  
3. **`run_supervisor(config)`**: Main control loop

   * Input: Supervisor configuration  
   * Output: SupervisorState with cycle history  
   * Logic: Iterates for configured cycles → calls agents in sequence based on config → tracks best hypothesis → detects stagnation → early stopping when no improvement

**Stagnation Detection**: Tracks top-ranked hypothesis across cycles → stops early if leader doesn't change for `no_improve_patience` cycles

**Outputs**:

* `ℹ️ Creating stub for missing agent...`: Confirms using placeholder functions  
* `🔄 Cycle 1 / 3...`: Shows supervisor main loop in action  
* `✅ Supervisor finished: No improvement`: Early stopping triggered (stubs always produce same "best" hypothesis)  
* `📊 Cycle summary`: Final summary showing consistent results across cycles

**High-Level Logic**: Acts as the "executive controller" that manages the overall scientific discovery process. This stubbed version validates the control flow logic before full system integration.

---

### **Cell 10: Full System Integration `[GpGDbkmqv2Of]`**

**Purpose**: Grand finale demonstrating the complete, end-to-end AI Co-Scientist system with all real agents working together.

**Key Components**:

1. **`CorrectedGenerationAgent`**: Improved version with better prompts for generating multiple distinct hypotheses

2. **Agent Integration Functions**:

   * **`_run_..._step` Functions**: Adapter functions that call real agents and integrate results into `FullSystemState`  
   * **Robust Ranking Adapter**: Handles cases with fewer than two hypotheses to prevent crashes  
3. **`IntegratedSupervisor`**: Main orchestration class

   * **Reordered Logic Flow**: **Generate → Proximity → Reflect → Rank → Evolve → (Re-Reflect & Re-Rank)**  
   * Handles real-world API errors and rate limiting  
   * Tracks comprehensive state across all cycles  
4. **`test_full_system()`**: Main demonstration function

   * Sets real research goal: liver fibrosis epigenetic therapy  
   * Configures system parameters  
   * Runs complete multi-cycle workflow  
   * Generates comprehensive final report

**Real-World Error Handling**:

* **Network Issues**: `ERROR:__main__:Web academic search failed...` (DuckDuckGo timeouts)  
* **API Rate Limiting**: `ResourceExhausted: 429 You exceeded your current quota...` (Google Gemini API limits)  
* **Graceful Degradation**: System continues operation using fallback mechanisms

**Cycle Execution**:

* **Cycle 1**: Generation → literature search → reflection → ranking → evolution → re-evaluation  
* **Cycle 2**: Continued refinement with evolved hypotheses → more API rate limiting → system resilience demonstrated  
* **Convergence**: System produces final ranked hypotheses with complete lineage tracking

**Final Deliverables**:

* Top-ranked hypothesis from complete research process  
* Final meta-review feedback and system insights  
* Complete cycle history and evolution tracking  
* Comprehensive research report ready for human review

**High-Level Logic**: Simulates a complete AI-driven scientific research process, demonstrating the system's ability to autonomously conduct research from literature review through hypothesis generation, evaluation, and iterative improvement, handling real-world challenges and producing actionable scientific insights.

