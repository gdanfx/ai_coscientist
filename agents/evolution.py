"""
Evolution Agent - Multi-Strategy Hypothesis Improvement

This module implements the EvolutionAgent that applies multiple strategies
to improve and evolve hypotheses using various creativity techniques.
"""

import re
import random
import statistics
import logging
from typing import List, Dict, Any, Optional, Tuple

from core.data_structures import (
    EvolutionStrategy, EvolutionStep, EvolvedHypothesis, EvolutionState,
    HypothesisReview, RankingState, ReflectionState,
    create_timestamp, create_hypothesis_id
)
from utils.llm_client import get_global_llm_client

logger = logging.getLogger(__name__)


class EvolutionAgent:
    """
    Evolution Agent that applies multiple strategies to improve and evolve hypotheses.
    
    Strategies include:
    - Simplification: Focus and reduce complexity
    - Combination: Merge complementary hypotheses
    - Analogical Reasoning: Apply cross-domain insights
    - Radical Variation: Explore paradigm-shifting approaches
    - Constraint Relaxation: Expand possibility space
    - Domain Transfer: Adapt successful patterns from other fields
    """

    def __init__(self, llm_client=None):
        self.llm = llm_client or get_global_llm_client()
        self.generation_counter = 0
        self.evolution_history = []

    def simplify_hypothesis(self, hypothesis_review: HypothesisReview,
                          ranking_info: Dict[str, Any]) -> EvolutionStep:
        """Strategy 1: Simplify and focus the hypothesis."""
        logger.info(f"Applying simplification strategy to {hypothesis_review.hypothesis_id}")

        simplification_prompt = f"""You are a strategic scientific advisor specializing in hypothesis refinement. Your task is to SIMPLIFY and FOCUS this research hypothesis to make it more achievable and impactful.

ORIGINAL HYPOTHESIS ({hypothesis_review.hypothesis_id}):
{hypothesis_review.hypothesis_text}

CURRENT PERFORMANCE:
- Reflection Score: {hypothesis_review.overall_score:.2f}/10
- Tournament Ranking: #{ranking_info.get('rank', 'Unknown')}
- Key Strengths: {'; '.join(hypothesis_review.strengths[:3])}
- Key Weaknesses: {'; '.join(hypothesis_review.weaknesses[:3])}

SIMPLIFICATION STRATEGY:
Your goal is to create a more focused, achievable version that:
1. Reduces complexity while maintaining core innovation
2. Focuses on the most promising aspect
3. Makes the approach more feasible and testable
4. Maintains or increases potential impact

EVOLVED HYPOTHESIS:
[Provide the simplified, focused version of the hypothesis]

IMPROVEMENT RATIONALE:
[Explain specifically how this simplification improves the original - 2-3 sentences]

EXPECTED BENEFITS:
- [Benefit 1: How feasibility improves]
- [Benefit 2: How testability improves]
- [Benefit 3: How focus improves impact]

POTENTIAL RISKS:
- [Risk 1: What might be lost in simplification]
- [Risk 2: Any new limitations introduced]

PREDICTED SCORE CHANGES (-5 to +5):
- Novelty Change: [score change and brief reason]
- Feasibility Change: [score change and brief reason]
- Impact Change: [score change and brief reason]

CONFIDENCE: [1-10 in this improvement]"""

        try:
            response = self.llm.invoke(simplification_prompt)
            if response.error:
                logger.error(f"LLM error during simplification: {response.error}")
                return self._create_fallback_evolution_step([hypothesis_review.hypothesis_id], EvolutionStrategy.SIMPLIFICATION)
                
            return self._parse_evolution_response(
                response.content, [hypothesis_review.hypothesis_id],
                EvolutionStrategy.SIMPLIFICATION
            )
        except Exception as e:
            logger.error(f"Simplification failed for {hypothesis_review.hypothesis_id}: {e}")
            return self._create_fallback_evolution_step([hypothesis_review.hypothesis_id], EvolutionStrategy.SIMPLIFICATION)

    def combine_hypotheses(self, hypothesis_reviews: List[HypothesisReview],
                          ranking_infos: List[Dict[str, Any]]) -> EvolutionStep:
        """Strategy 2: Combine multiple top hypotheses."""
        hypothesis_ids = [hr.hypothesis_id for hr in hypothesis_reviews]
        logger.info(f"Applying combination strategy to {', '.join(hypothesis_ids)}")

        # Prepare hypothesis summaries
        hypothesis_summaries = []
        for i, (review, ranking) in enumerate(zip(hypothesis_reviews, ranking_infos)):
            summary = f"""
HYPOTHESIS {i+1} ({review.hypothesis_id}):
Content: {review.hypothesis_text[:300]}...
Strengths: {'; '.join(review.strengths[:2])}
Ranking: #{ranking.get('rank', 'Unknown')} (Score: {review.overall_score:.1f}/10)"""
            hypothesis_summaries.append(summary)

        combination_prompt = f"""You are an innovative scientific researcher expert at synthesizing breakthrough ideas. Your task is to COMBINE the best elements from these top-performing hypotheses into a superior unified approach.

TOP HYPOTHESES TO COMBINE:
{''.join(hypothesis_summaries)}

COMBINATION STRATEGY:
Create a novel hybrid hypothesis that:
1. Integrates the strongest elements from each hypothesis
2. Creates synergistic effects between different approaches
3. Eliminates redundancies and weaknesses
4. Results in greater impact than individual components
5. Maintains feasibility while increasing innovation

COMBINED EVOLVED HYPOTHESIS:
[Present the unified hypothesis that combines the best elements]

INTEGRATION RATIONALE:
[Explain how you combined the hypotheses and why this synthesis is superior - 3-4 sentences]

SYNERGISTIC BENEFITS:
- [Benefit 1: How combination creates new capabilities]
- [Benefit 2: How different approaches reinforce each other]
- [Benefit 3: How combined approach increases impact]
- [Benefit 4: How integration reduces individual limitations]

POTENTIAL INTEGRATION RISKS:
- [Risk 1: Complexity from combination]
- [Risk 2: Potential conflicts between approaches]

PREDICTED SCORE CHANGES (-5 to +5):
- Novelty Change: [score change - combinations often increase novelty]
- Feasibility Change: [score change - may decrease due to complexity]
- Impact Change: [score change - typically increases significantly]

CONFIDENCE: [1-10 in this combination approach]"""

        try:
            response = self.llm.invoke(combination_prompt)
            if response.error:
                logger.error(f"LLM error during combination: {response.error}")
                return self._create_fallback_evolution_step(hypothesis_ids, EvolutionStrategy.COMBINATION)
                
            return self._parse_evolution_response(
                response.content, hypothesis_ids, EvolutionStrategy.COMBINATION
            )
        except Exception as e:
            logger.error(f"Combination failed for {hypothesis_ids}: {e}")
            return self._create_fallback_evolution_step(hypothesis_ids, EvolutionStrategy.COMBINATION)

    def analogical_reasoning(self, hypothesis_review: HypothesisReview,
                           ranking_info: Dict[str, Any]) -> EvolutionStep:
        """Strategy 3: Apply analogical reasoning from other successful domains."""
        logger.info(f"Applying analogical reasoning to {hypothesis_review.hypothesis_id}")

        analogical_prompt = f"""You are a cross-disciplinary innovation expert skilled at applying successful patterns from one domain to create breakthroughs in another. Your task is to evolve this hypothesis using ANALOGICAL REASONING from other successful scientific domains.

ORIGINAL HYPOTHESIS ({hypothesis_review.hypothesis_id}):
{hypothesis_review.hypothesis_text}

CURRENT PERFORMANCE:
- Reflection Score: {hypothesis_review.overall_score:.2f}/10
- Tournament Ranking: #{ranking_info.get('rank', 'Unknown')}
- Strengths: {'; '.join(hypothesis_review.strengths[:3])}
- Areas for improvement: {'; '.join(hypothesis_review.weaknesses[:2])}

ANALOGICAL REASONING STRATEGY:
1. Identify successful patterns from other domains (immunology, engineering, computer science, ecology, etc.)
2. Find analogous challenges that have been solved elegantly elsewhere
3. Adapt those successful approaches to this biomedical context
4. Create novel solutions inspired by cross-domain insights

EVOLVED HYPOTHESIS WITH ANALOGICAL INSPIRATION:
[Present the evolved hypothesis incorporating cross-domain insights]

ANALOGICAL INSPIRATION:
[Describe the specific analogy/domain that inspired this evolution - 2-3 sentences]

CROSS-DOMAIN BENEFITS:
- [Benefit 1: What successful pattern was adapted]
- [Benefit 2: How this analogy solves current limitations]
- [Benefit 3: What new capabilities this enables]
- [Benefit 4: How this increases novelty and innovation]

ADAPTATION RISKS:
- [Risk 1: Challenges in domain transfer]
- [Risk 2: Potential misalignment with biomedical context]

PREDICTED SCORE CHANGES (-5 to +5):
- Novelty Change: [score change - analogical reasoning typically increases novelty significantly]
- Feasibility Change: [score change - depends on analogy complexity]
- Impact Change: [score change - cross-domain insights often increase impact]

CONFIDENCE: [1-10 in this analogical approach]"""

        try:
            response = self.llm.invoke(analogical_prompt)
            if response.error:
                logger.error(f"LLM error during analogical reasoning: {response.error}")
                return self._create_fallback_evolution_step([hypothesis_review.hypothesis_id], EvolutionStrategy.ANALOGICAL_REASONING)
                
            return self._parse_evolution_response(
                response.content, [hypothesis_review.hypothesis_id],
                EvolutionStrategy.ANALOGICAL_REASONING
            )
        except Exception as e:
            logger.error(f"Analogical reasoning failed for {hypothesis_review.hypothesis_id}: {e}")
            return self._create_fallback_evolution_step([hypothesis_review.hypothesis_id], EvolutionStrategy.ANALOGICAL_REASONING)

    def radical_variation(self, hypothesis_review: HypothesisReview,
                         ranking_info: Dict[str, Any]) -> EvolutionStep:
        """Strategy 4: Create radical variations exploring very different approaches."""
        logger.info(f"Applying radical variation to {hypothesis_review.hypothesis_id}")

        radical_prompt = f"""You are a visionary scientific researcher known for paradigm-shifting breakthroughs. Your task is to create a RADICAL VARIATION of this hypothesis that explores a completely different approach while addressing the same fundamental problem.

ORIGINAL HYPOTHESIS ({hypothesis_review.hypothesis_id}):
{hypothesis_review.hypothesis_text}

CURRENT APPROACH ANALYSIS:
- Current Score: {hypothesis_review.overall_score:.2f}/10
- Current Strengths: {'; '.join(hypothesis_review.strengths[:2])}
- Current Limitations: {'; '.join(hypothesis_review.weaknesses[:2])}

RADICAL VARIATION STRATEGY:
Create a fundamentally different approach that:
1. Addresses the same core problem from a completely new angle
2. Challenges conventional assumptions in the field
3. Explores cutting-edge technologies or methodologies
4. Has potential for paradigm-shifting impact
5. May be higher risk but also higher reward

Think beyond incremental improvements - imagine a breakthrough that would make current approaches obsolete.

RADICAL EVOLVED HYPOTHESIS:
[Present the radically different approach to the same problem]

PARADIGM SHIFT RATIONALE:
[Explain what fundamental assumptions you're challenging and why this radical approach could be superior - 3-4 sentences]

BREAKTHROUGH POTENTIAL:
- [Revolutionary aspect 1: What paradigm this challenges]
- [Revolutionary aspect 2: What new possibilities this opens]
- [Revolutionary aspect 3: How this could transform the field]
- [Revolutionary aspect 4: What conventional limitations this bypasses]

HIGH-RISK FACTORS:
- [Risk 1: Technical challenges of radical approach]
- [Risk 2: Acceptance and validation difficulties]
- [Risk 3: Resource and timeline implications]

PREDICTED SCORE CHANGES (-5 to +5):
- Novelty Change: [score change - should be strongly positive for radical innovation]
- Feasibility Change: [score change - may decrease due to cutting-edge nature]
- Impact Change: [score change - potential for transformative impact]

CONFIDENCE: [1-10 in this radical approach - may be lower due to higher uncertainty]"""

        try:
            response = self.llm.invoke(radical_prompt)
            if response.error:
                logger.error(f"LLM error during radical variation: {response.error}")
                return self._create_fallback_evolution_step([hypothesis_review.hypothesis_id], EvolutionStrategy.RADICAL_VARIATION)
                
            return self._parse_evolution_response(
                response.content, [hypothesis_review.hypothesis_id],
                EvolutionStrategy.RADICAL_VARIATION
            )
        except Exception as e:
            logger.error(f"Radical variation failed for {hypothesis_review.hypothesis_id}: {e}")
            return self._create_fallback_evolution_step([hypothesis_review.hypothesis_id], EvolutionStrategy.RADICAL_VARIATION)

    def constraint_relaxation(self, hypothesis_review: HypothesisReview,
                            ranking_info: Dict[str, Any],
                            original_constraints: List[str]) -> EvolutionStep:
        """Strategy 5: Relax constraints to explore new possibilities."""
        logger.info(f"Applying constraint relaxation to {hypothesis_review.hypothesis_id}")

        constraint_prompt = f"""You are a strategic research planner exploring expanded possibilities. Your task is to evolve this hypothesis by RELAXING KEY CONSTRAINTS to unlock new potential approaches.

ORIGINAL HYPOTHESIS ({hypothesis_review.hypothesis_id}):
{hypothesis_review.hypothesis_text}

CURRENT CONSTRAINTS:
{chr(10).join([f"- {constraint}" for constraint in original_constraints])}

CURRENT PERFORMANCE:
- Score: {hypothesis_review.overall_score:.2f}/10
- Main limitations: {'; '.join(hypothesis_review.weaknesses[:3])}

CONSTRAINT RELAXATION STRATEGY:
1. Identify which constraints are most limiting the hypothesis potential
2. Explore what becomes possible if we relax 1-2 key constraints
3. Develop evolved approach that leverages this expanded possibility space
4. Maintain core innovation while expanding scope

EVOLVED HYPOTHESIS WITH RELAXED CONSTRAINTS:
[Present the evolved hypothesis with expanded possibilities]

CONSTRAINTS RELAXED:
[Specify which 1-2 constraints you relaxed and why]

EXPANDED POSSIBILITIES:
- [Possibility 1: What new approaches become available]
- [Possibility 2: What additional impact becomes possible]
- [Possibility 3: What technological opportunities open up]
- [Possibility 4: How timeline or scope can be optimized]

TRADE-OFF ANALYSIS:
- [Trade-off 1: What becomes more challenging]
- [Trade-off 2: What additional resources might be needed]

PREDICTED SCORE CHANGES (-5 to +5):
- Novelty Change: [score change and reasoning]
- Feasibility Change: [score change - may decrease with relaxed constraints]
- Impact Change: [score change - typically increases with expanded scope]

CONFIDENCE: [1-10 in this constraint relaxation approach]"""

        try:
            response = self.llm.invoke(constraint_prompt)
            if response.error:
                logger.error(f"LLM error during constraint relaxation: {response.error}")
                return self._create_fallback_evolution_step([hypothesis_review.hypothesis_id], EvolutionStrategy.CONSTRAINT_RELAXATION)
                
            return self._parse_evolution_response(
                response.content, [hypothesis_review.hypothesis_id],
                EvolutionStrategy.CONSTRAINT_RELAXATION
            )
        except Exception as e:
            logger.error(f"Constraint relaxation failed for {hypothesis_review.hypothesis_id}: {e}")
            return self._create_fallback_evolution_step([hypothesis_review.hypothesis_id], EvolutionStrategy.CONSTRAINT_RELAXATION)

    def domain_transfer(self, hypothesis_review: HypothesisReview,
                       ranking_info: Dict[str, Any]) -> EvolutionStep:
        """Strategy 6: Transfer successful patterns from other domains."""
        logger.info(f"Applying domain transfer to {hypothesis_review.hypothesis_id}")

        domain_prompt = f"""You are an expert in cross-disciplinary research patterns. Your task is to evolve this hypothesis by transferring successful methodologies, technologies, or approaches from other scientific domains.

ORIGINAL HYPOTHESIS ({hypothesis_review.hypothesis_id}):
{hypothesis_review.hypothesis_text}

CURRENT PERFORMANCE:
- Score: {hypothesis_review.overall_score:.2f}/10
- Strengths: {'; '.join(hypothesis_review.strengths[:2])}
- Improvement areas: {'; '.join(hypothesis_review.weaknesses[:2])}

DOMAIN TRANSFER STRATEGY:
1. Identify successful methodologies from other fields (materials science, AI/ML, physics, etc.)
2. Find transferable technologies or approaches that could enhance this hypothesis
3. Adapt these external innovations to the current research context
4. Create hybrid approaches that leverage cross-domain expertise

EVOLVED HYPOTHESIS WITH DOMAIN TRANSFER:
[Present the hypothesis enhanced with cross-domain innovations]

DOMAIN TRANSFER SOURCE:
[Describe what domain/technology you're transferring from and why it's relevant]

CROSS-DOMAIN ENHANCEMENTS:
- [Enhancement 1: What technology/method was transferred]
- [Enhancement 2: How it improves the original approach]
- [Enhancement 3: What new capabilities it enables]
- [Enhancement 4: How it addresses current limitations]

TRANSFER CHALLENGES:
- [Challenge 1: Technical adaptation requirements]
- [Challenge 2: Compatibility with existing approach]

PREDICTED SCORE CHANGES (-5 to +5):
- Novelty Change: [score change - domain transfer often increases novelty]
- Feasibility Change: [score change - depends on transfer complexity]
- Impact Change: [score change - cross-domain innovations often increase impact]

CONFIDENCE: [1-10 in this domain transfer approach]"""

        try:
            response = self.llm.invoke(domain_prompt)
            if response.error:
                logger.error(f"LLM error during domain transfer: {response.error}")
                return self._create_fallback_evolution_step([hypothesis_review.hypothesis_id], EvolutionStrategy.DOMAIN_TRANSFER)
                
            return self._parse_evolution_response(
                response.content, [hypothesis_review.hypothesis_id],
                EvolutionStrategy.DOMAIN_TRANSFER
            )
        except Exception as e:
            logger.error(f"Domain transfer failed for {hypothesis_review.hypothesis_id}: {e}")
            return self._create_fallback_evolution_step([hypothesis_review.hypothesis_id], EvolutionStrategy.DOMAIN_TRANSFER)

    def _parse_evolution_response(self, response_text: str, parent_ids: List[str],
                                strategy: EvolutionStrategy) -> EvolutionStep:
        """Parse LLM evolution response into structured format."""
        try:
            # Extract evolved hypothesis content
            content_patterns = [
                r'evolved hypothesis.*?:(.*?)(?:improvement rationale|rationale|benefits)',
                r'hypothesis.*?:(.*?)(?:improvement|rationale|benefits)',
                r'combined.*?hypothesis.*?:(.*?)(?:integration|rationale|benefits)'
            ]

            evolved_content = "Evolution content could not be extracted"
            for pattern in content_patterns:
                match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
                if match:
                    evolved_content = match.group(1).strip()[:1000]
                    break

            # Extract rationale
            rationale_patterns = [
                r'rationale.*?:(.*?)(?:benefits|expected|predicted)',
                r'improvement.*?rationale.*?:(.*?)(?:benefits|expected|predicted)',
                r'integration.*?rationale.*?:(.*?)(?:benefits|expected|predicted)'
            ]

            improvement_rationale = "Rationale could not be extracted"
            for pattern in rationale_patterns:
                match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
                if match:
                    improvement_rationale = match.group(1).strip()[:500]
                    break

            # Extract benefits and risks
            benefits = self._extract_list_items(response_text, ["benefit", "benefits", "advantages"])
            risks = self._extract_list_items(response_text, ["risk", "risks", "challenge", "challenges"])

            # Extract predicted score changes
            score_changes = self._extract_score_changes(response_text)

            # Extract confidence
            confidence_match = re.search(r'confidence.*?(\d+(?:\.\d+)?)', response_text, re.IGNORECASE)
            confidence = float(confidence_match.group(1)) if confidence_match else 7.0

            step_id = create_hypothesis_id(f"evolution_{strategy.value}")

            return EvolutionStep(
                step_id=step_id,
                parent_hypothesis_ids=parent_ids,
                strategy_used=strategy,
                evolution_prompt="Evolution prompt executed",
                evolved_content=evolved_content,
                improvement_rationale=improvement_rationale,
                expected_benefits=benefits,
                potential_risks=risks,
                novelty_increase=score_changes.get('novelty', 0.0),
                feasibility_change=score_changes.get('feasibility', 0.0),
                impact_increase=score_changes.get('impact', 0.0),
                confidence=confidence,
                timestamp=create_timestamp()
            )

        except Exception as e:
            logger.error(f"Failed to parse evolution response: {e}")
            return self._create_fallback_evolution_step(parent_ids, strategy)

    def _extract_list_items(self, text: str, keywords: List[str]) -> List[str]:
        """Extract bulleted list items from text."""
        items = []
        for keyword in keywords:
            pattern = f'{keyword}.*?:(.*?)(?:[A-Z]{{2,}}.*?:|$)'
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                section_text = match.group(1)
                lines = section_text.split('\n')
                for line in lines:
                    line = line.strip()
                    if re.match(r'^[-•*]\s+', line) or re.match(r'^\d+\.\s+', line):
                        cleaned_line = re.sub(r'^[-•*\d.]\s*', '', line)
                        if len(cleaned_line) > 10:
                            items.append(cleaned_line[:150])
                break

        return items[:4] if items else [f"No specific {keywords[0]} identified"]

    def _extract_score_changes(self, text: str) -> Dict[str, float]:
        """Extract predicted score changes from text."""
        changes = {}
        criteria = ['novelty', 'feasibility', 'impact']

        for criterion in criteria:
            pattern = f'{criterion}.*?change.*?([+-]?\d+(?:\.\d+)?)'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    changes[criterion] = float(match.group(1))
                except ValueError:
                    changes[criterion] = 0.0
            else:
                changes[criterion] = 0.0

        return changes

    def _create_fallback_evolution_step(self, parent_ids: List[str],
                                      strategy: EvolutionStrategy) -> EvolutionStep:
        """Create fallback evolution step when parsing fails."""
        step_id = create_hypothesis_id(f"fallback_{strategy.value}")

        return EvolutionStep(
            step_id=step_id,
            parent_hypothesis_ids=parent_ids,
            strategy_used=strategy,
            evolution_prompt="Fallback evolution step",
            evolved_content=f"Evolved hypothesis using {strategy.value} strategy (fallback generated)",
            improvement_rationale=f"Applied {strategy.value} strategy to improve the original hypothesis",
            expected_benefits=[f"Expected benefits from {strategy.value}", "Improved hypothesis quality"],
            potential_risks=["Fallback evolution may need manual refinement"],
            novelty_increase=0.5,
            feasibility_change=0.0,
            impact_increase=0.5,
            confidence=5.0,
            timestamp=create_timestamp()
        )

    def evolve_top_hypotheses(self, ranking_state: RankingState,
                            reflection_state: ReflectionState,
                            original_constraints: List[str] = None,
                            top_n: int = 3) -> EvolutionState:
        """Evolve the top N hypotheses using multiple strategies."""
        logger.info(f"Starting evolution of top {top_n} hypotheses")

        if original_constraints is None:
            original_constraints = ["Must be clinically applicable", "Focus on translational approaches"]

        evolution_state = EvolutionState()
        self.generation_counter += 1

        # Get top hypotheses from ranking
        top_rankings = ranking_state.final_rankings[:top_n]
        review_lookup = {r.hypothesis_id: r for r in reflection_state.hypothesis_reviews}

        # Apply different strategies to different hypotheses
        for i, ranking in enumerate(top_rankings):
            hypothesis_id = ranking['hypothesis_id']
            hypothesis_review = review_lookup.get(hypothesis_id)
            if not hypothesis_review:
                logger.warning(f"Could not find review for hypothesis {hypothesis_id}, skipping evolution.")
                continue

            logger.info(f"Evolving #{ranking['rank']}: {hypothesis_id}")

            # Strategy selection based on rank and characteristics
            evolution_steps = []

            # Rank 1: Apply multiple strategies
            if ranking['rank'] == 1:
                # Simplify the top hypothesis for broader applicability
                evolution_steps.append(
                    self.simplify_hypothesis(hypothesis_review, ranking)
                )
                # Also create a radical variation for breakthrough potential
                evolution_steps.append(
                    self.radical_variation(hypothesis_review, ranking)
                )

            # Rank 2: Focus on enhancement strategies
            elif ranking['rank'] == 2:
                # Apply analogical reasoning for novel insights
                evolution_steps.append(
                    self.analogical_reasoning(hypothesis_review, ranking)
                )
                # Try domain transfer for cross-domain innovations
                evolution_steps.append(
                    self.domain_transfer(hypothesis_review, ranking)
                )

            # Rank 3: Apply targeted improvements
            else:
                # Simplify to increase feasibility
                evolution_steps.append(
                    self.simplify_hypothesis(hypothesis_review, ranking)
                )
                # Relax constraints for expanded possibilities
                evolution_steps.append(
                    self.constraint_relaxation(hypothesis_review, ranking, original_constraints)
                )

            # Create evolved hypotheses from each evolution step
            for step in evolution_steps:
                evolved_hyp = self._create_evolved_hypothesis(step, hypothesis_review)
                evolution_state.evolved_hypotheses.append(evolved_hyp)

        # Apply combination strategy if we have multiple top hypotheses
        if len(top_rankings) >= 2:
            top_2_reviews = [review_lookup[ranking['hypothesis_id']] for ranking in top_rankings[:2] if ranking['hypothesis_id'] in review_lookup]
            top_2_rankings = top_rankings[:2]

            if len(top_2_reviews) >= 2:
                combination_step = self.combine_hypotheses(top_2_reviews, top_2_rankings)
                combined_hyp = self._create_evolved_hypothesis(combination_step, None)
                evolution_state.evolved_hypotheses.append(combined_hyp)

        # Generate analytics and summary
        evolution_state.evolution_statistics = self._compute_evolution_statistics(evolution_state)
        evolution_state.strategy_effectiveness = self._analyze_strategy_effectiveness(evolution_state)
        evolution_state.best_evolved_hypothesis = self._identify_best_evolved(evolution_state)
        evolution_state.generation_summary = self._generate_evolution_summary(evolution_state)

        logger.info(f"Evolution completed: {len(evolution_state.evolved_hypotheses)} evolved hypotheses generated")
        return evolution_state

    def _create_evolved_hypothesis(self, evolution_step: EvolutionStep,
                                 original_review: Optional[HypothesisReview]) -> EvolvedHypothesis:
        """Create evolved hypothesis from evolution step."""
        hypothesis_id = create_hypothesis_id(f"evolved_{evolution_step.strategy_used.value}")

        # Predict new scores based on evolution step
        if original_review:
            predicted_scores = {
                'novelty': min(10, max(1, original_review.criteria.novelty_score + evolution_step.novelty_increase)),
                'feasibility': min(10, max(1, original_review.criteria.feasibility_score + evolution_step.feasibility_change)),
                'scientific_rigor': original_review.criteria.scientific_rigor_score,  # Usually preserved
                'impact_potential': min(10, max(1, original_review.criteria.impact_potential_score + evolution_step.impact_increase)),
                'testability': original_review.criteria.testability_score  # Usually preserved
            }
        else:
            # For combinations, estimate higher scores
            predicted_scores = {
                'novelty': 8.0 + evolution_step.novelty_increase,
                'feasibility': 7.0 + evolution_step.feasibility_change,
                'scientific_rigor': 7.5,
                'impact_potential': 8.5 + evolution_step.impact_increase,
                'testability': 7.0
            }

        # Generate implementation roadmap
        roadmap = self._generate_implementation_roadmap(evolution_step)
        competitive_advantages = evolution_step.expected_benefits[:3]

        return EvolvedHypothesis(
            hypothesis_id=hypothesis_id,
            evolved_content=evolution_step.evolved_content,
            parent_hypothesis_ids=evolution_step.parent_hypothesis_ids,
            evolution_lineage=[evolution_step],
            generation_number=self.generation_counter,
            predicted_scores=predicted_scores,
            improvement_summary=evolution_step.improvement_rationale,
            competitive_advantages=competitive_advantages,
            implementation_roadmap=roadmap
        )

    def _generate_implementation_roadmap(self, evolution_step: EvolutionStep) -> List[str]:
        """Generate implementation roadmap based on evolution strategy."""
        strategy = evolution_step.strategy_used

        roadmap_templates = {
            EvolutionStrategy.SIMPLIFICATION: [
                "Phase 1: Validate simplified approach in proof-of-concept studies",
                "Phase 2: Optimize focused methodology for maximum impact",
                "Phase 3: Scale up simplified approach for broader application"
            ],
            EvolutionStrategy.COMBINATION: [
                "Phase 1: Validate synergistic effects between combined approaches",
                "Phase 2: Optimize integration protocols and workflows",
                "Phase 3: Implement unified approach in comprehensive studies"
            ],
            EvolutionStrategy.ANALOGICAL_REASONING: [
                "Phase 1: Validate cross-domain adaptation in target context",
                "Phase 2: Refine approach based on domain-specific requirements",
                "Phase 3: Demonstrate superior performance vs conventional methods"
            ],
            EvolutionStrategy.RADICAL_VARIATION: [
                "Phase 1: Establish proof-of-concept for paradigm-shifting approach",
                "Phase 2: Address technical challenges and validation requirements",
                "Phase 3: Develop breakthrough approach for transformative impact"
            ],
            EvolutionStrategy.CONSTRAINT_RELAXATION: [
                "Phase 1: Explore expanded possibilities with relaxed constraints",
                "Phase 2: Optimize approach within new possibility space",
                "Phase 3: Demonstrate enhanced impact from expanded scope"
            ],
            EvolutionStrategy.DOMAIN_TRANSFER: [
                "Phase 1: Adapt transferred technology to current research context",
                "Phase 2: Validate cross-domain integration and compatibility",
                "Phase 3: Optimize hybrid approach for maximum effectiveness"
            ]
        }

        return roadmap_templates.get(strategy, [
            "Phase 1: Validate evolved approach",
            "Phase 2: Optimize implementation",
            "Phase 3: Scale for maximum impact"
        ])

    def _compute_evolution_statistics(self, evolution_state: EvolutionState) -> Dict[str, float]:
        """Compute comprehensive evolution statistics."""
        if not evolution_state.evolved_hypotheses:
            return {}

        predicted_overall_scores = []
        novelty_improvements = []
        impact_improvements = []
        feasibility_changes = []
        confidence_levels = []

        for evolved_hyp in evolution_state.evolved_hypotheses:
            # Calculate predicted overall score
            scores = evolved_hyp.predicted_scores
            overall = (scores['novelty'] * 0.25 + scores['feasibility'] * 0.20 +
                      scores['scientific_rigor'] * 0.25 + scores['impact_potential'] * 0.20 +
                      scores['testability'] * 0.10)
            predicted_overall_scores.append(overall)

            # Get evolution metrics from lineage
            if evolved_hyp.evolution_lineage:
                step = evolved_hyp.evolution_lineage[0]
                novelty_improvements.append(step.novelty_increase)
                impact_improvements.append(step.impact_increase)
                feasibility_changes.append(step.feasibility_change)
                confidence_levels.append(step.confidence)

        if not predicted_overall_scores:
            return {}

        return {
            'total_evolved_hypotheses': len(evolution_state.evolved_hypotheses),
            'mean_predicted_score': statistics.mean(predicted_overall_scores),
            'max_predicted_score': max(predicted_overall_scores),
            'min_predicted_score': min(predicted_overall_scores),
            'mean_novelty_improvement': statistics.mean(novelty_improvements),
            'mean_impact_improvement': statistics.mean(impact_improvements),
            'mean_feasibility_change': statistics.mean(feasibility_changes),
            'mean_evolution_confidence': statistics.mean(confidence_levels),
            'strategies_applied': len(set(hyp.evolution_lineage[0].strategy_used for hyp in evolution_state.evolved_hypotheses)),
            'generation_number': self.generation_counter
        }

    def _analyze_strategy_effectiveness(self, evolution_state: EvolutionState) -> Dict[str, float]:
        """Analyze effectiveness of different evolution strategies."""
        strategy_metrics = {}

        for evolved_hyp in evolution_state.evolved_hypotheses:
            if evolved_hyp.evolution_lineage:
                step = evolved_hyp.evolution_lineage[0]
                strategy = step.strategy_used.value

                if strategy not in strategy_metrics:
                    strategy_metrics[strategy] = {
                        'count': 0,
                        'total_novelty_gain': 0,
                        'total_impact_gain': 0,
                        'total_confidence': 0,
                        'predicted_scores': []
                    }

                metrics = strategy_metrics[strategy]
                metrics['count'] += 1
                metrics['total_novelty_gain'] += step.novelty_increase
                metrics['total_impact_gain'] += step.impact_increase
                metrics['total_confidence'] += step.confidence

                # Calculate predicted overall score
                scores = evolved_hyp.predicted_scores
                overall = (scores['novelty'] * 0.25 + scores['feasibility'] * 0.20 +
                          scores['scientific_rigor'] * 0.25 + scores['impact_potential'] * 0.20 +
                          scores['testability'] * 0.10)
                metrics['predicted_scores'].append(overall)

        # Calculate effectiveness scores
        effectiveness = {}
        for strategy, metrics in strategy_metrics.items():
            if metrics['count'] > 0:
                effectiveness[f"{strategy}_avg_novelty_gain"] = metrics['total_novelty_gain'] / metrics['count']
                effectiveness[f"{strategy}_avg_impact_gain"] = metrics['total_impact_gain'] / metrics['count']
                effectiveness[f"{strategy}_avg_confidence"] = metrics['total_confidence'] / metrics['count']
                effectiveness[f"{strategy}_avg_predicted_score"] = statistics.mean(metrics['predicted_scores'])
                effectiveness[f"{strategy}_applications"] = metrics['count']

        return effectiveness

    def _identify_best_evolved(self, evolution_state: EvolutionState) -> Optional[str]:
        """Identify the best evolved hypothesis based on predicted scores."""
        if not evolution_state.evolved_hypotheses:
            return None

        best_hypothesis = max(
            evolution_state.evolved_hypotheses,
            key=lambda h: statistics.mean(list(h.predicted_scores.values()))
        )

        return best_hypothesis.hypothesis_id

    def _generate_evolution_summary(self, evolution_state: EvolutionState) -> str:
        """Generate comprehensive evolution summary."""
        stats = evolution_state.evolution_statistics
        strategy_eff = evolution_state.strategy_effectiveness
        best_id = evolution_state.best_evolved_hypothesis

        # Find best hypothesis for details
        best_hyp = None
        if best_id:
            best_hyp = next((h for h in evolution_state.evolved_hypotheses if h.hypothesis_id == best_id), None)

        if not stats:
            return "Evolution summary could not be generated."

        summary = f"""
EVOLUTION AGENT GENERATION SUMMARY

🧬 EVOLUTION OVERVIEW:
• Generation Number: {stats.get('generation_number', 1)}
• Evolved Hypotheses: {stats.get('total_evolved_hypotheses', 0)}
• Strategies Applied: {stats.get('strategies_applied', 0)}
• Evolution Confidence: {stats.get('mean_evolution_confidence', 0):.1f}/10

📈 IMPROVEMENT METRICS:
• Mean Predicted Score: {stats.get('mean_predicted_score', 0):.2f}/10
• Score Range: {stats.get('min_predicted_score', 0):.1f} - {stats.get('max_predicted_score', 0):.1f}
• Average Novelty Gain: {stats.get('mean_novelty_improvement', 0):+.1f} points
• Average Impact Gain: {stats.get('mean_impact_improvement', 0):+.1f} points
• Feasibility Change: {stats.get('mean_feasibility_change', 0):+.1f} points

🎯 STRATEGY EFFECTIVENESS:
{self._format_strategy_effectiveness(strategy_eff)}

🏆 BEST EVOLVED HYPOTHESIS:
{self._format_best_hypothesis(best_hyp)}

🚀 IMPLEMENTATION RECOMMENDATIONS:
{self._generate_implementation_recommendations(evolution_state)}
"""

        return summary.strip()

    def _format_strategy_effectiveness(self, strategy_effectiveness: Dict[str, float]) -> str:
        """Format strategy effectiveness for display."""
        strategies = set()
        for key in strategy_effectiveness.keys():
            if '_avg_predicted_score' in key:
                strategies.add(key.replace('_avg_predicted_score', ''))

        formatted = ""
        for strategy in sorted(list(strategies)):
            score = strategy_effectiveness.get(f"{strategy}_avg_predicted_score", 0)
            novelty = strategy_effectiveness.get(f"{strategy}_avg_novelty_gain", 0)
            impact = strategy_effectiveness.get(f"{strategy}_avg_impact_gain", 0)
            count = strategy_effectiveness.get(f"{strategy}_applications", 0)

            formatted += f"\n• {strategy.title()}: Score {score:.1f} (+{novelty:.1f} novelty, +{impact:.1f} impact) [{int(count)} applications]"

        return formatted if formatted else "No strategy effectiveness data available"

    def _format_best_hypothesis(self, best_hyp: Optional[EvolvedHypothesis]) -> str:
        """Format best hypothesis for display."""
        if not best_hyp:
            return "No best hypothesis identified"

        strategy = best_hyp.evolution_lineage[0].strategy_used.value if best_hyp.evolution_lineage else "unknown"
        predicted_score = statistics.mean(list(best_hyp.predicted_scores.values()))

        return f"""
• ID: {best_hyp.hypothesis_id}
• Strategy Used: {strategy.title()}
• Predicted Score: {predicted_score:.2f}/10
• Key Advantages: {'; '.join(best_hyp.competitive_advantages[:2])}
• Parent(s): {', '.join(best_hyp.parent_hypothesis_ids)}"""

    def _generate_implementation_recommendations(self, evolution_state: EvolutionState) -> str:
        """Generate implementation recommendations."""
        recommendations = []
        stats = evolution_state.evolution_statistics

        if not stats:
            return "No recommendations available."

        # Best hypothesis recommendations
        if evolution_state.best_evolved_hypothesis:
            recommendations.append(f"• PRIORITY: Focus on {evolution_state.best_evolved_hypothesis} for detailed development")

        # Strategy-based recommendations
        mean_score = stats.get('mean_predicted_score', 0)
        if mean_score > 8.0:
            recommendations.append("• EXCELLENT: High-quality evolved hypotheses - proceed to validation")
        elif mean_score > 7.0:
            recommendations.append("• STRONG: Good evolution results - consider further refinement")
        else:
            recommendations.append("• ITERATE: Consider additional evolution rounds")

        # Novelty vs feasibility trade-offs
        novelty_gain = stats.get('mean_novelty_improvement', 0)
        feasibility_change = stats.get('mean_feasibility_change', 0)

        if novelty_gain > 1.5 and feasibility_change < -1.0:
            recommendations.append("• BALANCE: High novelty gained but feasibility decreased - validate technical approach")
        elif novelty_gain > 2.0:
            recommendations.append("• BREAKTHROUGH: Significant novelty increases - potential for high impact")

        # Next steps
        if len(evolution_state.evolved_hypotheses) >= 3:
            recommendations.append("• NEXT: Run reflection and ranking on evolved hypotheses for comparison")

        return '\n'.join(recommendations) if recommendations else "• Proceed with evolved hypotheses as planned"


# Factory function for easy instantiation
def create_evolution_agent(**kwargs) -> EvolutionAgent:
    """Factory function to create an evolution agent with optional parameters."""
    return EvolutionAgent(**kwargs)


# Integration function
def run_evolution_agent(ranking_state: RankingState, reflection_state: ReflectionState,
                       original_constraints: List[str] = None, top_n: int = 3) -> EvolutionState:
    """
    Integration function for the Evolution Agent.
    
    Args:
        ranking_state: State object with final_rankings
        reflection_state: State object with hypothesis_reviews
        original_constraints: List of original research constraints
        top_n: Number of top hypotheses to evolve
        
    Returns:
        EvolutionState with evolved hypotheses
    """
    logger.info("🧬 Engaging Evolution Agent...")
    
    if not hasattr(ranking_state, 'final_rankings') or not ranking_state.final_rankings:
        logger.warning("No rankings found in ranking_state for evolution")
        return EvolutionState()

    if not hasattr(reflection_state, 'hypothesis_reviews') or not reflection_state.hypothesis_reviews:
        logger.warning("No hypothesis reviews found in reflection_state for evolution")
        return EvolutionState()

    # Initialize and run the agent
    evolution_agent = EvolutionAgent()
    evolution_results = evolution_agent.evolve_top_hypotheses(
        ranking_state, reflection_state, original_constraints, top_n
    )

    logger.info(f"✅ Evolution Agent finished: {len(evolution_results.evolved_hypotheses)} hypotheses evolved")
    
    return evolution_results


# Testing function
def test_evolution_agent():
    """Test the Evolution Agent with sample data."""
    logger.info("Testing Evolution Agent...")
    
    # Import required structures
    from core.data_structures import (
        ReviewCriteria, HypothesisReview, ReflectionState, RankingState
    )
    
    # Create mock data for testing
    mock_reviews = []
    mock_rankings = []
    
    hypotheses_content = [
        "Novel epigenetic reprogramming approach targeting DNMT3A for liver fibrosis treatment",
        "AI-driven personalized medicine platform for optimized combination therapies",
        "Single-cell epigenomic mapping to identify therapeutic vulnerabilities"
    ]
    
    for i, content in enumerate(hypotheses_content):
        # Create review
        criteria = ReviewCriteria(
            novelty_score=7.5 + i * 0.3,
            feasibility_score=6.5 + i * 0.2,
            scientific_rigor_score=7.8 + i * 0.1,
            impact_potential_score=8.2 + i * 0.4,
            testability_score=7.0 + i * 0.2,
            novelty_reasoning="Novel approach",
            feasibility_reasoning="Technically feasible",
            scientific_rigor_reasoning="Well-grounded",
            impact_potential_reasoning="High impact",
            testability_reasoning="Clear validation"
        )
        
        review = HypothesisReview(
            hypothesis_id=f"test_hyp_{i+1}",
            hypothesis_text=content,
            criteria=criteria,
            overall_score=7.5 + i * 0.3,
            overall_assessment="Strong hypothesis with evolution potential",
            strengths=["Novel approach", "Clear rationale", "Strong foundation"],
            weaknesses=["Needs validation", "Resource requirements"],
            recommendations=["Further testing", "Optimize approach"],
            confidence_level=8.0,
            review_timestamp=create_timestamp(),
            reviewer_type="detailed"
        )
        mock_reviews.append(review)
        
        # Create ranking
        ranking = {
            'rank': i + 1,
            'hypothesis_id': f"test_hyp_{i+1}",
            'final_elo_rating': 1300 - (i * 50),
            'win_rate': 75 - (i * 10)
        }
        mock_rankings.append(ranking)
    
    # Create mock states
    mock_reflection_state = ReflectionState()
    mock_reflection_state.hypothesis_reviews = mock_reviews
    
    mock_ranking_state = RankingState()
    mock_ranking_state.final_rankings = mock_rankings
    
    try:
        # Test agent
        agent = EvolutionAgent(llm_client=None)  # Use None to trigger fallback
        result = agent.evolve_top_hypotheses(
            mock_ranking_state, 
            mock_reflection_state, 
            ["Must be clinically applicable", "Focus on translational approaches"],
            top_n=3
        )
        
        # Validate results
        assert len(result.evolved_hypotheses) > 0
        assert result.evolution_statistics is not None
        assert isinstance(result.strategy_effectiveness, dict)
        assert result.generation_summary is not None
        
        logger.info(f"Evolution Agent test completed successfully: "
                   f"{len(result.evolved_hypotheses)} hypotheses evolved, "
                   f"mean predicted score: {result.evolution_statistics.get('mean_predicted_score', 0):.2f}/10")
        return result
        
    except Exception as e:
        logger.error(f"Evolution Agent test failed: {e}")
        raise


if __name__ == "__main__":
    # Run test if executed directly
    test_result = test_evolution_agent()
    print(f"Test completed: {len(test_result.evolved_hypotheses)} hypotheses evolved, "
          f"best hypothesis: {test_result.best_evolved_hypothesis}")