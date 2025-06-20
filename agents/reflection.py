"""
Robust Reflection Agent - Automated Peer Review System

This module implements the RobustReflectionAgent that conducts automated peer review
of hypotheses using multiple scientific criteria with flexible LLM output parsing.
"""

import re
import logging
import statistics
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from core.data_structures import (
    ReviewCriteria, HypothesisReview, ReflectionState, 
    create_timestamp, validate_score
)
from utils.llm_client import get_global_llm_client

logger = logging.getLogger(__name__)


class RobustReflectionAgent:
    """
    Robust Reflection Agent that performs automated peer review of hypotheses.
    
    Features:
    - Two-tier review process (quick/detailed)
    - Five scientific criteria evaluation
    - Flexible LLM output parsing
    - Robust error handling and fallbacks
    - Comprehensive batch statistics
    """

    def __init__(self, llm_client=None):
        """Initialize the Reflection Agent with LLM client."""
        self.llm = llm_client or get_global_llm_client()
        
        # Criteria weights for overall score calculation
        self.criteria_weights = {
            'novelty_score': 0.25,
            'feasibility_score': 0.20,
            'scientific_rigor_score': 0.25,
            'impact_potential_score': 0.20,
            'testability_score': 0.10
        }
        
        logger.info("RobustReflectionAgent initialized")

    def quick_review(self, hypothesis_text: str, hypothesis_id: str) -> HypothesisReview:
        """Perform rapid evaluation using structured prompts."""
        logger.info(f"Performing quick review of hypothesis: {hypothesis_id}")

        quick_prompt = f"""You are a scientific peer reviewer. Please evaluate this hypothesis:

HYPOTHESIS: {hypothesis_text}

Please provide your evaluation in this format:

**SCORES (1-10 scale):**
Novelty: [score] - [brief reason]
Feasibility: [score] - [brief reason]
Scientific Rigor: [score] - [brief reason]
Impact Potential: [score] - [brief reason]
Testability: [score] - [brief reason]

**STRENGTHS:**
• [strength 1]
• [strength 2]
• [strength 3]

**WEAKNESSES:**
• [weakness 1]
• [weakness 2]
• [weakness 3]

**RECOMMENDATIONS:**
• [recommendation 1]
• [recommendation 2]
• [recommendation 3]

**OVERALL:** [2-3 sentence summary assessment]

**CONFIDENCE:** [1-10] in your review accuracy"""

        try:
            response = self.llm.invoke(quick_prompt)
            if response.error:
                logger.error(f"LLM error during quick review: {response.error}")
                return self._create_enhanced_fallback_review(hypothesis_text, hypothesis_id, "quick")
            
            return self._parse_flexible_review(response.content, hypothesis_text, hypothesis_id, "quick")

        except Exception as e:
            logger.error(f"Quick review failed for {hypothesis_id}: {e}")
            return self._create_enhanced_fallback_review(hypothesis_text, hypothesis_id, "quick")

    def detailed_review(self, hypothesis_text: str, hypothesis_id: str, research_goal: str = "") -> HypothesisReview:
        """Perform comprehensive evaluation for high-scoring hypotheses."""
        logger.info(f"Performing detailed review of hypothesis: {hypothesis_id}")

        detailed_prompt = f"""You are a senior scientific peer reviewer. Conduct a comprehensive evaluation of this hypothesis.

RESEARCH CONTEXT: {research_goal}

HYPOTHESIS: {hypothesis_text}

Please structure your review as follows:

**DETAILED EVALUATION:**

1. **Novelty (1-10):** [score]
   [Detailed assessment of originality and innovation]

2. **Feasibility (1-10):** [score]
   [Analysis of technical achievability and resource requirements]

3. **Scientific Rigor (1-10):** [score]
   [Evaluation of methodological soundness and evidence base]

4. **Impact Potential (1-10):** [score]
   [Assessment of significance and broader implications]

5. **Testability (1-10):** [score]
   [Analysis of experimental validation approaches]

**COMPREHENSIVE ASSESSMENT:**

**Major Strengths:**
• [detailed strength 1]
• [detailed strength 2]
• [detailed strength 3]
• [detailed strength 4]

**Key Concerns:**
• [detailed concern 1]
• [detailed concern 2]
• [detailed concern 3]
• [detailed concern 4]

**Improvement Recommendations:**
• [specific recommendation 1]
• [specific recommendation 2]
• [specific recommendation 3]
• [specific recommendation 4]

**Summary:** [3-4 sentence overall evaluation]

**Review Confidence:** [1-10]"""

        try:
            response = self.llm.invoke(detailed_prompt)
            if response.error:
                logger.error(f"LLM error during detailed review: {response.error}")
                return self._create_enhanced_fallback_review(hypothesis_text, hypothesis_id, "detailed")
            
            return self._parse_flexible_review(response.content, hypothesis_text, hypothesis_id, "detailed")

        except Exception as e:
            logger.error(f"Detailed review failed for {hypothesis_id}: {e}")
            return self._create_enhanced_fallback_review(hypothesis_text, hypothesis_id, "detailed")

    def adaptive_batch_review(self, hypotheses: List[Dict[str, Any]], research_goal: str = "",
                            quick_threshold: float = 6.5, detailed_threshold: float = 7.5) -> ReflectionState:
        """
        Main entry point for batch hypothesis review with adaptive strategy.
        
        Args:
            hypotheses: List of hypothesis dictionaries with 'id' and 'content' keys
            research_goal: Research context for detailed reviews
            quick_threshold: Score threshold for considering detailed review
            detailed_threshold: Score threshold for mandatory detailed review
            
        Returns:
            ReflectionState with complete review results and statistics
        """
        logger.info(f"Starting adaptive batch review of {len(hypotheses)} hypotheses")

        reflection_state = ReflectionState()

        for i, hypothesis in enumerate(hypotheses):
            hypothesis_id = hypothesis.get('id', f"hypothesis_{i+1}")
            hypothesis_text = hypothesis.get('content', str(hypothesis))

            # Always start with quick review
            quick_review = self.quick_review(hypothesis_text, hypothesis_id)

            # Decide on detailed review based on score and review quality
            if (quick_review.overall_score >= detailed_threshold and
                not quick_review.reviewer_type.endswith('_heuristic')):
                logger.info(f"Hypothesis {hypothesis_id} scored {quick_review.overall_score:.1f}, performing detailed review")
                detailed_review = self.detailed_review(hypothesis_text, hypothesis_id, research_goal)
                reflection_state.hypothesis_reviews.append(detailed_review)
            else:
                logger.info(f"Hypothesis {hypothesis_id} scored {quick_review.overall_score:.1f}, using quick review")
                reflection_state.hypothesis_reviews.append(quick_review)

        # Generate comprehensive analytics
        reflection_state.review_statistics = self._compute_robust_statistics(reflection_state.hypothesis_reviews)
        reflection_state.quality_flags = self._identify_robust_quality_flags(reflection_state.hypothesis_reviews)
        reflection_state.batch_summary = self._generate_robust_summary(reflection_state)

        logger.info(f"Batch review completed: {len(reflection_state.hypothesis_reviews)} reviews generated")
        return reflection_state

    def _parse_flexible_review(self, response_text: str, hypothesis_text: str,
                              hypothesis_id: str, review_type: str) -> HypothesisReview:
        """Parse LLM review response with flexible patterns to handle varied outputs."""
        logger.debug(f"Parsing review response for {hypothesis_id}")

        try:
            # Clean the response text
            cleaned_text = response_text.replace('*', '').replace('#', '').replace('`', '')

            # Extract scores using multiple pattern attempts
            scores = self._extract_scores_flexible(cleaned_text)

            # Extract confidence
            confidence = self._extract_confidence_flexible(cleaned_text)

            # Extract lists using flexible patterns
            strengths = self._extract_lists_flexible(cleaned_text, ["strength", "strengths", "positive", "advantages"])
            weaknesses = self._extract_lists_flexible(cleaned_text, ["weakness", "weaknesses", "concern", "concerns", "limitation", "limitations"])
            recommendations = self._extract_lists_flexible(cleaned_text, ["recommendation", "recommendations", "suggest", "improve"])

            # Extract overall assessment
            overall_assessment = self._extract_overall_flexible(cleaned_text)

            # Create criteria with extracted reasoning
            criteria = ReviewCriteria(
                novelty_score=scores.get('novelty', 6.0),
                feasibility_score=scores.get('feasibility', 6.0),
                scientific_rigor_score=scores.get('rigor', 6.0),
                impact_potential_score=scores.get('impact', 6.0),
                testability_score=scores.get('testability', 6.0),
                novelty_reasoning=f"Novelty assessment: {scores.get('novelty', 6.0)}/10",
                feasibility_reasoning=f"Feasibility assessment: {scores.get('feasibility', 6.0)}/10",
                scientific_rigor_reasoning=f"Scientific rigor assessment: {scores.get('rigor', 6.0)}/10",
                impact_potential_reasoning=f"Impact potential assessment: {scores.get('impact', 6.0)}/10",
                testability_reasoning=f"Testability assessment: {scores.get('testability', 6.0)}/10"
            )

            # Calculate weighted overall score
            overall_score = (
                scores.get('novelty', 6.0) * self.criteria_weights['novelty_score'] +
                scores.get('feasibility', 6.0) * self.criteria_weights['feasibility_score'] +
                scores.get('rigor', 6.0) * self.criteria_weights['scientific_rigor_score'] +
                scores.get('impact', 6.0) * self.criteria_weights['impact_potential_score'] +
                scores.get('testability', 6.0) * self.criteria_weights['testability_score']
            )

            return HypothesisReview(
                hypothesis_id=hypothesis_id,
                hypothesis_text=hypothesis_text[:500] + "..." if len(hypothesis_text) > 500 else hypothesis_text,
                criteria=criteria,
                overall_score=overall_score,
                overall_assessment=overall_assessment,
                strengths=strengths,
                weaknesses=weaknesses,
                recommendations=recommendations,
                confidence_level=confidence,
                review_timestamp=create_timestamp(),
                reviewer_type=review_type
            )

        except Exception as e:
            logger.error(f"Flexible parsing failed for {hypothesis_id}: {e}")
            # If all parsing fails, create enhanced fallback
            return self._create_enhanced_fallback_review(hypothesis_text, hypothesis_id, review_type)

    def _extract_scores_flexible(self, text: str) -> Dict[str, float]:
        """Extract scores using multiple flexible patterns."""
        scores = {}

        # Define multiple patterns for each criterion
        pattern_sets = {
            'novelty': [
                r'novelty.*?(\d+(?:\.\d+)?)',
                r'original.*?(\d+(?:\.\d+)?)',
                r'innovation.*?(\d+(?:\.\d+)?)',
                r'novel.*?(\d+(?:\.\d+)?)'
            ],
            'feasibility': [
                r'feasibility.*?(\d+(?:\.\d+)?)',
                r'feasible.*?(\d+(?:\.\d+)?)',
                r'achievable.*?(\d+(?:\.\d+)?)',
                r'practical.*?(\d+(?:\.\d+)?)'
            ],
            'rigor': [
                r'rigor.*?(\d+(?:\.\d+)?)',
                r'rigorous.*?(\d+(?:\.\d+)?)',
                r'scientific.*?rigor.*?(\d+(?:\.\d+)?)',
                r'methodological.*?(\d+(?:\.\d+)?)',
                r'evidence.*?(\d+(?:\.\d+)?)'
            ],
            'impact': [
                r'impact.*?(\d+(?:\.\d+)?)',
                r'significance.*?(\d+(?:\.\d+)?)',
                r'important.*?(\d+(?:\.\d+)?)',
                r'potential.*?(\d+(?:\.\d+)?)'
            ],
            'testability': [
                r'testability.*?(\d+(?:\.\d+)?)',
                r'testable.*?(\d+(?:\.\d+)?)',
                r'validation.*?(\d+(?:\.\d+)?)',
                r'experimental.*?(\d+(?:\.\d+)?)'
            ]
        }

        for criterion, patterns in pattern_sets.items():
            score_found = False
            for pattern in patterns:
                match = re.search(pattern, text.lower())
                if match:
                    try:
                        score = float(match.group(1))
                        if 1 <= score <= 10:  # Validate score range
                            scores[criterion] = validate_score(score)
                            score_found = True
                            break
                    except ValueError:
                        continue

            if not score_found:
                # Look for any number in proximity to criterion words
                criterion_word = criterion if criterion != 'rigor' else 'rigor'
                proximity_pattern = f'{criterion_word}.{{0,50}}?(\d+)'
                match = re.search(proximity_pattern, text.lower())
                if match:
                    try:
                        score = float(match.group(1))
                        scores[criterion] = validate_score(score)
                    except ValueError:
                        scores[criterion] = 6.0
                else:
                    scores[criterion] = 6.0

        return scores

    def _extract_confidence_flexible(self, text: str) -> float:
        """Extract confidence score using flexible patterns."""
        patterns = [
            r'confidence.*?(\d+(?:\.\d+)?)',
            r'confident.*?(\d+(?:\.\d+)?)',
            r'certainty.*?(\d+(?:\.\d+)?)',
            r'sure.*?(\d+(?:\.\d+)?)'
        ]

        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    confidence = float(match.group(1))
                    if 1 <= confidence <= 10:
                        return confidence
                except ValueError:
                    continue

        return 7.0  # Default confidence

    def _extract_lists_flexible(self, text: str, keywords: List[str]) -> List[str]:
        """Extract bulleted or numbered lists using flexible patterns."""
        items = []

        for keyword in keywords:
            # Find section containing the keyword
            lines = text.split('\n')
            section_started = False

            for i, line in enumerate(lines):
                if keyword.lower() in line.lower() and ':' in line:
                    section_started = True
                    continue

                if section_started:
                    line = line.strip()

                    # Stop if we hit another section header
                    if line and line.endswith(':') and any(word in line.lower() for word in ['score', 'assessment', 'evaluation', 'summary']):
                        break

                    # Extract bullet points or numbered items
                    if re.match(r'^[-•*]\s+', line) or re.match(r'^\d+\.\s+', line):
                        cleaned_line = re.sub(r'^[-•*\d.]\s*', '', line)
                        if len(cleaned_line) > 5:  # Ensure meaningful content
                            items.append(cleaned_line[:150])  # Limit length

                    # Also catch lines that start with text but are clearly list items
                    elif line and not line.startswith(('**', '#', 'Score', 'Assessment')):
                        if 10 < len(line) < 200:  # Reasonable length for list item
                            items.append(line[:150])

            if items:  # If we found items with this keyword, use them
                break

        # If no structured lists found, try to extract from general text
        if not items:
            # Look for sentences that contain action words or assessment language
            sentences = re.split(r'[.!?]+', text)
            for sentence in sentences:
                sentence = sentence.strip()
                if (any(keyword.lower() in sentence.lower() for keyword in keywords) and
                    len(sentence) > 15 and len(sentence) < 200):
                    items.append(sentence[:150])

        return items[:4] if items else [f"No specific {keywords[0]} identified"]

    def _extract_overall_flexible(self, text: str) -> str:
        """Extract overall assessment using flexible patterns."""
        keywords = ["overall", "summary", "assessment", "conclusion", "evaluation"]

        for keyword in keywords:
            # Look for keyword followed by colon
            pattern = f'{keyword}.*?:(.*?)(?:[A-Z]{{2,}}.*?:|$)'
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                assessment = match.group(1).strip()
                if len(assessment) > 20:
                    return assessment[:400] + "..." if len(assessment) > 400 else assessment

        # Fallback: look for any substantial paragraph
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
        if paragraphs:
            # Take the longest paragraph as likely assessment
            longest = max(paragraphs, key=len)
            return longest[:400] + "..." if len(longest) > 400 else longest

        return "Overall assessment could not be extracted from review."

    def _create_enhanced_fallback_review(self, hypothesis_text: str, hypothesis_id: str, review_type: str) -> HypothesisReview:
        """Create enhanced fallback review with heuristic scoring."""
        # Analyze hypothesis text to provide more realistic scores
        text_lower = hypothesis_text.lower()

        # Heuristic scoring based on content analysis
        novelty_score = 6.5 if any(word in text_lower for word in ['novel', 'new', 'innovative', 'breakthrough']) else 5.5
        feasibility_score = 7.0 if any(word in text_lower for word in ['demonstrated', 'established', 'proven', 'validated']) else 6.0
        rigor_score = 7.0 if any(word in text_lower for word in ['literature', 'evidence', 'studies', 'research']) else 5.5
        impact_score = 7.5 if any(word in text_lower for word in ['therapeutic', 'clinical', 'treatment', 'therapy']) else 6.0
        testability_score = 7.0 if any(word in text_lower for word in ['experiment', 'test', 'measure', 'assay']) else 6.0

        criteria = ReviewCriteria(
            novelty_score=novelty_score,
            feasibility_score=feasibility_score,
            scientific_rigor_score=rigor_score,
            impact_potential_score=impact_score,
            testability_score=testability_score,
            novelty_reasoning=f"Heuristic assessment based on text analysis: {novelty_score}/10",
            feasibility_reasoning=f"Heuristic assessment based on text analysis: {feasibility_score}/10",
            scientific_rigor_reasoning=f"Heuristic assessment based on text analysis: {rigor_score}/10",
            impact_potential_reasoning=f"Heuristic assessment based on text analysis: {impact_score}/10",
            testability_reasoning=f"Heuristic assessment based on text analysis: {testability_score}/10"
        )

        # Calculate weighted overall score
        overall_score = (
            novelty_score * self.criteria_weights['novelty_score'] +
            feasibility_score * self.criteria_weights['feasibility_score'] +
            rigor_score * self.criteria_weights['scientific_rigor_score'] +
            impact_score * self.criteria_weights['impact_potential_score'] +
            testability_score * self.criteria_weights['testability_score']
        )

        # Generate heuristic strengths and weaknesses
        strengths = ["Addresses important research question"]
        if "epigenetic" in text_lower:
            strengths.append("Focuses on promising epigenetic mechanisms")
        if "therapeutic" in text_lower or "treatment" in text_lower:
            strengths.append("Has clear therapeutic relevance")
        if "experiment" in text_lower:
            strengths.append("Includes experimental validation approach")

        weaknesses = ["Requires detailed expert review for accurate assessment"]
        if len(hypothesis_text) < 200:
            weaknesses.append("Limited detail in hypothesis description")
        if "novel" not in text_lower and "new" not in text_lower:
            weaknesses.append("Novelty could be better articulated")

        recommendations = ["Conduct detailed expert peer review"]
        if "literature" not in text_lower:
            recommendations.append("Strengthen literature foundation")
        if "experiment" not in text_lower:
            recommendations.append("Develop specific experimental protocols")

        return HypothesisReview(
            hypothesis_id=hypothesis_id,
            hypothesis_text=hypothesis_text[:300] + "..." if len(hypothesis_text) > 300 else hypothesis_text,
            criteria=criteria,
            overall_score=overall_score,
            overall_assessment=f"Automated heuristic assessment. The hypothesis shows promise with an overall score of {overall_score:.1f}/10. Manual expert review recommended for detailed evaluation.",
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            confidence_level=5.0,  # Medium confidence for heuristic assessment
            review_timestamp=create_timestamp(),
            reviewer_type=review_type + "_heuristic"
        )

    def _compute_robust_statistics(self, reviews: List[HypothesisReview]) -> Dict[str, float]:
        """Compute comprehensive statistics across all reviews."""
        if not reviews:
            return {'error': 'No reviews available'}

        overall_scores = [r.overall_score for r in reviews if r.overall_score is not None]
        confidence_scores = [r.confidence_level for r in reviews if r.confidence_level is not None]

        if not overall_scores:
            return {'error': 'No valid scores found'}

        criteria_scores = {
            'novelty': [r.criteria.novelty_score for r in reviews if r.criteria and r.criteria.novelty_score is not None],
            'feasibility': [r.criteria.feasibility_score for r in reviews if r.criteria and r.criteria.feasibility_score is not None],
            'rigor': [r.criteria.scientific_rigor_score for r in reviews if r.criteria and r.criteria.scientific_rigor_score is not None],
            'impact': [r.criteria.impact_potential_score for r in reviews if r.criteria and r.criteria.impact_potential_score is not None],
            'testability': [r.criteria.testability_score for r in reviews if r.criteria and r.criteria.testability_score is not None]
        }

        stats = {
            'mean_overall_score': statistics.mean(overall_scores),
            'median_overall_score': statistics.median(overall_scores),
            'std_overall_score': statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0,
            'max_overall_score': max(overall_scores),
            'min_overall_score': min(overall_scores),
            'mean_confidence': statistics.mean(confidence_scores) if confidence_scores else 5.0,
            'excellent_count': len([s for s in overall_scores if s >= 8.5]),
            'high_scoring_count': len([s for s in overall_scores if s >= 7.5]),
            'moderate_scoring_count': len([s for s in overall_scores if 5.0 <= s < 7.5]),
            'low_scoring_count': len([s for s in overall_scores if s < 5.0]),
            'detailed_review_count': len([r for r in reviews if r.reviewer_type == "detailed"]),
            'quick_review_count': len([r for r in reviews if r.reviewer_type == "quick"]),
            'heuristic_review_count': len([r for r in reviews if "heuristic" in r.reviewer_type])
        }

        # Add criteria-specific statistics safely
        for criterion, scores in criteria_scores.items():
            if scores:
                stats[f'mean_{criterion}'] = statistics.mean(scores)
                stats[f'max_{criterion}'] = max(scores)
                stats[f'min_{criterion}'] = min(scores)
            else:
                stats[f'mean_{criterion}'] = 6.0
                stats[f'max_{criterion}'] = 6.0
                stats[f'min_{criterion}'] = 6.0

        return stats

    def _identify_robust_quality_flags(self, reviews: List[HypothesisReview]) -> List[str]:
        """Identify quality flags with robust error handling."""
        flags = []

        if not reviews:
            flags.append("NO_REVIEWS_GENERATED")
            return flags

        # Count review types
        heuristic_count = len([r for r in reviews if "heuristic" in r.reviewer_type])
        total_count = len(reviews)

        if heuristic_count == total_count:
            flags.append("ALL_HEURISTIC_REVIEWS")
        elif heuristic_count > total_count * 0.5:
            flags.append("MAJORITY_HEURISTIC_REVIEWS")

        # Check scores safely
        overall_scores = [r.overall_score for r in reviews if r.overall_score is not None]
        confidence_scores = [r.confidence_level for r in reviews if r.confidence_level is not None]

        if overall_scores:
            mean_score = statistics.mean(overall_scores)
            if mean_score < 5.0:
                flags.append("LOW_AVERAGE_QUALITY")
            elif mean_score >= 8.0:
                flags.append("HIGH_QUALITY_BATCH")

        if confidence_scores:
            mean_confidence = statistics.mean(confidence_scores)
            if mean_confidence < 6.0:
                flags.append("LOW_REVIEWER_CONFIDENCE")
            elif mean_confidence >= 8.5:
                flags.append("HIGH_REVIEWER_CONFIDENCE")

        return flags

    def _generate_robust_summary(self, reflection_state: ReflectionState) -> str:
        """Generate comprehensive batch summary."""
        stats = reflection_state.review_statistics
        flags = reflection_state.quality_flags
        reviews = reflection_state.hypothesis_reviews

        if not reviews:
            return "No reviews were generated. Please check the input data and try again."

        # Safe access to statistics
        total_reviews = len(reviews)
        mean_score = stats.get('mean_overall_score', 0)
        mean_confidence = stats.get('mean_confidence', 0)

        summary = f"""
ROBUST REFLECTION AGENT BATCH SUMMARY

📊 REVIEW OVERVIEW:
• Total hypotheses reviewed: {total_reviews}
• Heuristic reviews: {stats.get('heuristic_review_count', 0)}
• Quick reviews: {stats.get('quick_review_count', 0)}
• Detailed reviews: {stats.get('detailed_review_count', 0)}

📈 PERFORMANCE METRICS:
• Mean overall score: {mean_score:.2f}/10
• Score range: {stats.get('min_overall_score', 0):.1f} - {stats.get('max_overall_score', 0):.1f}
• Reviewer confidence: {mean_confidence:.1f}/10

🎯 SCORE DISTRIBUTION:
• Excellent (≥8.5): {stats.get('excellent_count', 0)} hypotheses
• High (7.5-8.4): {stats.get('high_scoring_count', 0)} hypotheses
• Moderate (5.0-7.4): {stats.get('moderate_scoring_count', 0)} hypotheses
• Low (<5.0): {stats.get('low_scoring_count', 0)} hypotheses

⚠️ QUALITY INDICATORS: {', '.join(flags) if flags else 'None detected'}

🏆 TOP HYPOTHESIS: {self._get_top_hypothesis_summary(reviews)}

💡 NEXT STEPS:
{self._generate_next_steps_recommendations(stats, flags)}
"""

        return summary.strip()

    def _get_top_hypothesis_summary(self, reviews: List[HypothesisReview]) -> str:
        """Safely get top hypothesis summary."""
        if not reviews:
            return "No reviews available"

        valid_reviews = [r for r in reviews if r.overall_score is not None]
        if not valid_reviews:
            return "No valid scores available"

        top_review = max(valid_reviews, key=lambda r: r.overall_score)
        return f"{top_review.hypothesis_id} (Score: {top_review.overall_score:.2f}/10)"

    def _generate_next_steps_recommendations(self, stats: Dict[str, float], flags: List[str]) -> str:
        """Generate actionable next steps based on review results."""
        recommendations = []

        mean_score = stats.get('mean_overall_score', 0)
        heuristic_count = stats.get('heuristic_review_count', 0)

        # Primary recommendations based on review quality
        if "ALL_HEURISTIC_REVIEWS" in flags:
            recommendations.append("• CRITICAL: All reviews were heuristic - improve LLM prompt parsing")
        elif "MAJORITY_HEURISTIC_REVIEWS" in flags:
            recommendations.append("• IMPROVE: Majority heuristic reviews - enhance LLM response structure")

        # Score-based recommendations
        if mean_score >= 7.5:
            recommendations.append("• PROCEED: High quality hypotheses - ready for Ranking Agent")
        elif mean_score >= 6.0:
            recommendations.append("• CONSIDER: Moderate quality - may benefit from Evolution Agent")
        else:
            recommendations.append("• REVISE: Low quality scores - return to Generation Agent")

        # Confidence-based recommendations
        if "LOW_REVIEWER_CONFIDENCE" in flags:
            recommendations.append("• VALIDATE: Low confidence suggests need for human expert review")

        # Default recommendation
        if not recommendations:
            recommendations.append("• CONTINUE: Proceed with current workflow")

        return '\n'.join(recommendations)


# Factory function for easy instantiation
def create_reflection_agent(**kwargs) -> RobustReflectionAgent:
    """Factory function to create a reflection agent with optional parameters."""
    return RobustReflectionAgent(**kwargs)


# Integration function 
def run_reflection_agent(generation_state, research_goal: str = "") -> ReflectionState:
    """
    Integration function for the Reflection Agent.
    
    Args:
        generation_state: State object with generated_proposals
        research_goal: Research context for detailed reviews
        
    Returns:
        ReflectionState with review results
    """
    logger.info("🔬 Engaging Reflection Agent...")
    
    # Get proposals from generation state
    proposals = getattr(generation_state, 'generated_proposals', [])
    if not proposals:
        logger.warning("No proposals found in generation_state for reflection")
        return ReflectionState()

    # Initialize and run the agent
    reflection_agent = RobustReflectionAgent()
    reflection_results = reflection_agent.adaptive_batch_review(proposals, research_goal)

    logger.info(f"✅ Reflection Agent finished: {len(reflection_results.hypothesis_reviews)} reviews completed")
    
    return reflection_results


# Testing function
def test_reflection_agent():
    """Test the Robust Reflection Agent with sample hypotheses."""
    logger.info("Testing Robust Reflection Agent...")
    
    # Test hypotheses with different characteristics
    test_hypotheses = [
        {
            "id": "hyp_epigenetic",
            "content": """Novel Hypothesis: Epigenetic Reprogramming of Hepatic Stellate Cells

We hypothesize that targeted inhibition of DNMT3A in activated hepatic stellate cells will reverse pathological methylation patterns and restore their quiescent phenotype, leading to resolution of liver fibrosis. This innovative approach leverages the reversible nature of epigenetic modifications.

Scientific Rationale:
- DNMT3A upregulation is consistently demonstrated in fibrotic liver tissue
- Hypermethylation of anti-fibrotic genes like PPAR-γ correlates with stellate cell activation
- Epigenetic targets offer therapeutic advantages due to reversibility

Experimental Validation:
1. In vitro DNMT3A knockdown in activated human stellate cells
2. Genome-wide methylation profiling using bisulfite sequencing
3. In vivo delivery using stellate cell-specific nanoparticles
4. Assessment of fibrosis markers and liver function recovery

Expected Impact: Could lead to first-in-class epigenetic therapy for liver fibrosis with potential for clinical translation within 3-5 years."""
        },
        {
            "id": "hyp_ai_personalized", 
            "content": """Breakthrough Hypothesis: AI-Driven Personalized Epigenetic Therapy

We propose developing a machine learning platform that analyzes individual patient epigenetic profiles to predict optimal combination therapies targeting multiple epigenetic enzymes (DNMT, HDAC, BRD4) for personalized liver fibrosis treatment.

Innovation: This represents the first AI-guided approach to epigenetic combination therapy, addressing the critical need for personalized medicine in liver fibrosis treatment.

Technical Approach:
- Multi-omics profiling of fibrosis patients (methylome, transcriptome, clinical data)
- Deep learning algorithms for treatment response prediction
- Validation in patient-derived organoid models
- Clinical pilot study with AI-recommended therapies

Transformative Potential: Could revolutionize liver fibrosis treatment by enabling precision medicine approaches with significantly improved patient outcomes."""
        }
    ]

    try:
        # Test agent
        agent = RobustReflectionAgent()
        result = agent.adaptive_batch_review(test_hypotheses, "Discover novel epigenetic targets for liver fibrosis treatment")

        # Validate results
        assert len(result.hypothesis_reviews) == len(test_hypotheses)
        assert result.review_statistics is not None
        assert isinstance(result.quality_flags, list)
        assert result.batch_summary is not None
        
        logger.info(f"Reflection Agent test completed successfully: "
                   f"{len(result.hypothesis_reviews)} reviews, "
                   f"mean score: {result.review_statistics.get('mean_overall_score', 0):.2f}")
        return result

    except Exception as e:
        logger.error(f"Reflection Agent test failed: {e}")
        raise


if __name__ == "__main__":
    # Run test if executed directly
    test_result = test_reflection_agent()
    print(f"Test completed: {len(test_result.hypothesis_reviews)} reviews generated, "
          f"mean confidence: {test_result.review_statistics.get('mean_confidence', 0):.1f}/10")