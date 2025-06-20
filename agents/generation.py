"""
Enhanced Multi-Source Literature Search Agent

This module implements the Generation Agent that searches multiple literature sources
and generates novel hypotheses based on synthesized research findings.
"""

import time
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from utils.llm_client import get_global_llm_client, LLMResponse
from utils.literature_search import multi_source_literature_search
from core.data_structures import GenerationState, create_hypothesis_id, create_timestamp

logger = logging.getLogger(__name__)


class EnhancedGenerationAgent:
    """
    Enhanced Generation Agent with multi-source literature search capabilities.
    
    This agent orchestrates the complete workflow:
    1. Generate optimized search queries
    2. Search multiple literature sources
    3. Synthesize knowledge from findings
    4. Generate novel hypotheses
    """

    def __init__(self, llm_client=None):
        """Initialize the Generation Agent with LLM client."""
        self.llm = llm_client or get_global_llm_client()
        logger.info("EnhancedGenerationAgent initialized")

    def generate_search_queries(self, state: GenerationState) -> GenerationState:
        """Generate optimized search queries for multiple sources."""
        logger.info("Generating multi-source search queries...")

        prompt = f"""Generate 3-4 simple, effective search queries for academic literature about: {state.research_goal}

Consider these constraints: {', '.join(state.constraints)}

Create queries that work well across academic databases (PubMed, ArXiv, CrossRef):
- Use simple keyword combinations 
- Avoid complex Boolean operators (AND, OR) with quotes
- Focus on core concepts and synonyms
- Each query should target a different aspect of the research goal

Format your response as a numbered list of simple keyword phrases:
1. [Simple keyword query 1]
2. [Simple keyword query 2] 
3. [Simple keyword query 3]
4. [Simple keyword query 4]

Example format:
1. CAR T cell therapy biomarkers
2. chimeric antigen receptor diagnostics
3. personalized cancer immunotherapy
4. cellular therapy companion diagnostics"""

        try:
            response = self.llm.invoke(prompt)
            
            if response.error:
                logger.error(f"LLM error during query generation: {response.error}")
                # Fallback queries
                base_terms = state.research_goal.lower().split()[:3]
                state.search_queries = [
                    ' '.join(base_terms),
                    f"{base_terms[0]} treatment" if len(base_terms) > 0 else "treatment",
                    f"{base_terms[0]} therapy" if len(base_terms) > 0 else "therapy",
                    "clinical trial"
                ]
            else:
                # Parse queries from LLM response
                queries = []
                lines = response.content.split('\n')

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    # Remove markdown formatting and extract query
                    line = line.replace('*', '').replace('_', '')

                    # Look for numbered items
                    if any(line.startswith(str(i) + '.') for i in range(1, 10)):
                        # Extract everything after the number and period
                        query = line.split('.', 1)[1].strip() if '.' in line else line
                        
                        # Remove database prefixes (PubMed:, ArXiv:, CrossRef:)
                        for prefix in ['PubMed:', 'ArXiv:', 'CrossRef:', 'PubMed (alternative, more specific):']:
                            if prefix in query:
                                query = query.split(prefix, 1)[1].strip()
                        
                        # Remove markdown code blocks and backticks
                        query = query.replace('`', '').strip()
                        
                        # Remove parentheses and extra formatting
                        query = query.replace('(', '').replace(')', '').strip()

                        if query and len(query) > 3:
                            queries.append(query)

                # Fallback if parsing failed
                if not queries:
                    base_terms = state.research_goal.lower().split()[:3]
                    queries = [
                        ' '.join(base_terms),
                        f"{base_terms[0]} treatment" if len(base_terms) > 0 else "treatment",
                        f"{base_terms[0]} therapy" if len(base_terms) > 0 else "therapy",
                        "clinical trial"
                    ]

                state.search_queries = queries[:4]

        except Exception as e:
            logger.error(f"Query generation failed: {e}")
            # Emergency fallback
            state.search_queries = [
                state.research_goal,
                f"{state.research_goal} treatment",
                f"{state.research_goal} therapy"
            ]

        state.status = "queries_generated"
        logger.info(f"Generated {len(state.search_queries)} optimized queries")
        
        return state

    def literature_exploration(self, state: GenerationState) -> GenerationState:
        """Search literature using multiple sources."""
        logger.info("Searching multiple literature sources...")

        try:
            findings = multi_source_literature_search(
                queries=state.search_queries,
                max_results_per_source=3,
                total_max_results=12
            )
            
            state.literature_findings = findings
            state.status = "literature_explored"

            # Log source breakdown
            source_counts = {}
            for finding in findings:
                source = finding['source']
                source_counts[source] = source_counts.get(source, 0) + 1

            # Always show all 3 sources that were searched, even if some returned 0 results
            all_sources = ["PubMed", "ArXiv", "CrossRef"]
            sources_with_results = [src for src in all_sources if src in source_counts]
            
            logger.info(f"Found {len(findings)} papers from {len(sources_with_results)}/{len(all_sources)} sources")
            for source in all_sources:
                count = source_counts.get(source, 0)
                if count > 0:
                    logger.info(f"  • {source}: {count} papers")
                else:
                    logger.info(f"  • {source}: 0 papers (no results found)")

        except Exception as e:
            logger.error(f"Literature search failed: {e}")
            state.literature_findings = []
            state.status = "literature_failed"
            raise RuntimeError(f"Literature search failed: {e}")

        return state

    def synthesize_knowledge(self, state: GenerationState) -> GenerationState:
        """Synthesize knowledge from multiple sources."""
        logger.info("Synthesizing knowledge from multiple sources...")

        if not state.literature_findings:
            logger.warning("No literature findings to synthesize")
            state.synthesized_knowledge = "No literature findings available for synthesis."
            state.status = "knowledge_synthesized"
            return state

        # Create structured findings text for LLM
        findings_text = self._format_literature_findings(state.literature_findings)

        prompt = f"""Synthesize this diverse research for: {state.research_goal}

Literature from Multiple Sources:
{findings_text}

Provide a comprehensive synthesis that:
1. Identifies key themes and patterns across sources
2. Highlights important findings and methodologies
3. Notes gaps or limitations in current research
4. Sets the foundation for novel hypothesis generation

Focus on synthesizing insights that could lead to breakthrough hypotheses."""

        try:
            response = self.llm.invoke(prompt)
            
            if response.error:
                logger.error(f"Knowledge synthesis failed: {response.error}")
                state.synthesized_knowledge = f"Knowledge synthesis encountered an error. Based on {len(state.literature_findings)} papers from multiple sources."
            else:
                state.synthesized_knowledge = response.content

        except Exception as e:
            logger.error(f"Knowledge synthesis failed: {e}")
            state.synthesized_knowledge = f"Knowledge synthesis encountered an error. Based on {len(state.literature_findings)} papers from multiple sources."

        state.status = "knowledge_synthesized"
        logger.info("Multi-source knowledge synthesis complete")
        
        return state

    def generate_hypotheses(self, state: GenerationState) -> GenerationState:
        """Generate novel hypotheses based on synthesized knowledge."""
        logger.info("Generating hypotheses from multi-source research...")

        if not state.synthesized_knowledge:
            logger.warning("No synthesized knowledge available for hypothesis generation")
            state.generated_proposals = []
            state.status = "hypotheses_generated"
            return state

        prompt = f"""Based on this multi-source research synthesis, generate 2-3 novel and distinct hypotheses for: {state.research_goal}

Constraints: {', '.join(state.constraints)}

Multi-Source Knowledge Base:
{state.synthesized_knowledge}

Generate hypotheses that:
1. Are novel and not directly stated in the literature
2. Build upon multiple sources of evidence
3. Are testable and feasible
4. Have potential for significant impact

Structure your response with each distinct hypothesis separated by '---'. Each hypothesis must be a clear, testable statement.

Example format:
Hypothesis 1: [Clear, testable hypothesis statement]
---
Hypothesis 2: [Clear, testable hypothesis statement]
---
Hypothesis 3: [Clear, testable hypothesis statement]"""

        try:
            response = self.llm.invoke(prompt)
            
            if response.error:
                logger.error(f"Hypothesis generation failed: {response.error}")
                hypothesis_content = "Hypothesis generation encountered an error."
            else:
                hypothesis_content = response.content

            # Parse multiple hypotheses from response
            proposals = self._parse_hypotheses_from_response(
                hypothesis_content, state
            )

        except Exception as e:
            logger.error(f"Hypothesis generation failed: {e}")
            proposals = [{
                "id": create_hypothesis_id("error"),
                "content": "Hypothesis generation encountered an error.",
                "timestamp": create_timestamp(),
                "based_on_sources": len(state.literature_findings),
                "source_diversity": len(set(f['source'] for f in state.literature_findings)) if state.literature_findings else 0
            }]

        state.generated_proposals = proposals
        state.status = "hypotheses_generated"
        state.iteration += 1

        source_count = len(state.literature_findings) if state.literature_findings else 0
        source_diversity = len(set(f['source'] for f in state.literature_findings)) if state.literature_findings else 0

        logger.info(f"Generated {len(proposals)} hypotheses based on {source_count} papers from {source_diversity} sources")
        
        return state

    def _format_literature_findings(self, findings: List[Dict[str, Any]]) -> str:
        """Format literature findings for LLM consumption."""
        if not findings:
            return "No literature findings available."

        formatted_findings = []
        for i, finding in enumerate(findings[:10], 1):  # Limit to top 10
            title = finding.get('title', 'No title')
            source = finding.get('source', 'Unknown')
            authors = finding.get('authors', ['Unknown'])
            year = finding.get('year', 'Unknown')
            key_findings = finding.get('key_findings', [])
            
            # Format authors (first 3)
            author_str = ', '.join(authors[:3])
            if len(authors) > 3:
                author_str += ' et al.'
            
            formatted = f"{i}. {title}\n"
            formatted += f"   Authors: {author_str} ({year})\n"
            formatted += f"   Source: {source}\n"
            
            if key_findings:
                formatted += f"   Key Findings: {'; '.join(key_findings[:2])}\n"
            
            formatted_findings.append(formatted)

        return '\n'.join(formatted_findings)

    def _parse_hypotheses_from_response(self, response_content: str, state: GenerationState) -> List[Dict[str, Any]]:
        """Parse multiple hypotheses from LLM response."""
        import re
        
        proposals = []
        
        # Split by '---' separator
        hypothesis_parts = [h.strip() for h in re.split(r'\n---\n|---', response_content) if h.strip()]
        
        for i, hypothesis_text in enumerate(hypothesis_parts):
            # Clean up hypothesis text
            # Remove "Hypothesis N:" prefix if present
            clean_text = re.sub(r'^Hypothesis\s+\d+:\s*', '', hypothesis_text, flags=re.IGNORECASE)
            clean_text = clean_text.strip()
            
            if clean_text and len(clean_text) > 20:  # Ensure meaningful content
                proposals.append({
                    "id": create_hypothesis_id(f"gen_{state.iteration}_{i+1}"),
                    "content": clean_text,
                    "timestamp": create_timestamp(),
                    "based_on_sources": len(state.literature_findings),
                    "source_diversity": len(set(f['source'] for f in state.literature_findings)) if state.literature_findings else 0
                })

        # Fallback: if no hypotheses parsed, create one from the full response
        if not proposals:
            proposals.append({
                "id": create_hypothesis_id(f"gen_{state.iteration}_fallback"),
                "content": response_content[:500] + "..." if len(response_content) > 500 else response_content,
                "timestamp": create_timestamp(),
                "based_on_sources": len(state.literature_findings),
                "source_diversity": len(set(f['source'] for f in state.literature_findings)) if state.literature_findings else 0
            })

        return proposals

    def run_complete_workflow(self, research_goal: str, constraints: List[str] = None) -> GenerationState:
        """
        Run the complete literature search and hypothesis generation workflow.
        
        This method provides a unified interface for running the entire workflow
        as specified in the agent interface contract.
        
        Args:
            research_goal: The research goal to investigate
            constraints: List of constraints to consider
            
        Returns:
            GenerationState with populated fields for search queries, literature findings,
            synthesized knowledge, and generated proposals
        """
        # Initialize state
        state = GenerationState(
            research_goal=research_goal,
            constraints=constraints or []
        )

        try:
            logger.info("Starting complete literature search workflow...")

            # Step 1: Generate search queries
            state = self.generate_search_queries(state)

            # Step 2: Search literature
            state = self.literature_exploration(state)

            # Step 3: Synthesize knowledge
            state = self.synthesize_knowledge(state)

            # Step 4: Generate hypotheses
            state = self.generate_hypotheses(state)

            logger.info("Complete workflow finished successfully")
            return state

        except Exception as e:
            logger.error(f"Workflow failed at step: {state.status}")
            logger.error(f"Error: {e}")

            # Return state with error information
            state.status = f"failed_at_{state.status}"
            state.generated_proposals = [{
                "id": create_hypothesis_id("workflow_error"),
                "content": f"Workflow failed during {state.status}: {str(e)}",
                "timestamp": create_timestamp(),
                "based_on_sources": len(state.literature_findings),
                "source_diversity": len(set(f['source'] for f in state.literature_findings)) if state.literature_findings else 0
            }]

            return state


# Factory function for easy instantiation
def create_generation_agent(**kwargs) -> EnhancedGenerationAgent:
    """Factory function to create a generation agent with optional parameters."""
    return EnhancedGenerationAgent(**kwargs)


# Testing function
def test_generation_agent():
    """Test the Enhanced Generation Agent with a sample research goal."""
    logger.info("Testing Enhanced Generation Agent...")
    
    # Test inputs
    test_research_goal = "Develop AI-driven personalized medicine approaches for neurodegenerative diseases"
    test_constraints = [
        "Must integrate multiple data modalities",
        "Focus on translational applications",
        "Consider ethical implications"
    ]

    try:
        # Create and run agent
        agent = EnhancedGenerationAgent()
        result = agent.run_complete_workflow(test_research_goal, test_constraints)

        # Validate results
        assert hasattr(result, 'search_queries')
        assert hasattr(result, 'literature_findings')
        assert hasattr(result, 'synthesized_knowledge')
        assert hasattr(result, 'generated_proposals')
        
        logger.info("Enhanced Generation Agent test completed successfully")
        return result

    except Exception as e:
        logger.error(f"Enhanced Generation Agent test failed: {e}")
        raise


if __name__ == "__main__":
    # Run test if executed directly
    test_result = test_generation_agent()
    print(f"Test completed with {len(test_result.generated_proposals)} proposals generated")