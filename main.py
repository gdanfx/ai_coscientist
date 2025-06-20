#!/usr/bin/env python3
"""
AI Co-Scientist - Main Entry Point

Command-line interface for running the complete AI Co-Scientist research system.
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.config import get_config
from core.data_structures import SupervisorConfig
from agents.supervisor import IntegratedSupervisor, SupervisorFactory, run_ai_coscientist_research


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration."""
    level = getattr(logging, log_level.upper())
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AI Co-Scientist - Automated Scientific Research System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Develop novel treatments for Alzheimer's disease"
  %(prog)s "AI applications in drug discovery" --max-cycles 5
  %(prog)s "Personalized medicine approaches" --constraints "Must be clinically applicable" "Focus on biomarkers"
  %(prog)s --test  # Run system test
        """
    )
    
    # Main research goal
    parser.add_argument(
        "research_goal",
        nargs="?",
        help="The research goal or question to investigate"
    )
    
    # Configuration options
    parser.add_argument(
        "--constraints",
        nargs="*",
        default=[],
        help="Research constraints (can specify multiple)"
    )
    
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=3,
        help="Maximum number of research cycles (default: 3)"
    )
    
    parser.add_argument(
        "--evolution-every",
        type=int,
        default=2,
        help="Run evolution every N cycles (default: 2)"
    )
    
    parser.add_argument(
        "--proximity-every", 
        type=int,
        default=1,
        help="Run proximity deduplication every N cycles (default: 1)"
    )
    
    parser.add_argument(
        "--meta-every",
        type=int, 
        default=2,
        help="Run meta-analysis every N cycles (default: 2)"
    )
    
    parser.add_argument(
        "--no-improve-patience",
        type=int,
        default=2,
        help="Stop early if no improvement for N cycles (default: 2)"
    )
    
    # Preset configurations
    parser.add_argument(
        "--rapid",
        action="store_true",
        help="Use rapid prototyping configuration (2 cycles, frequent evolution)"
    )
    
    parser.add_argument(
        "--thorough",
        action="store_true", 
        help="Use thorough research configuration (5 cycles, comprehensive analysis)"
    )
    
    # System options
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run system test instead of research"
    )
    
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate configuration and exit"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--log-file",
        help="Write logs to file (in addition to console)"
    )
    
    parser.add_argument(
        "--output",
        help="Save research results to JSON file"
    )
    
    return parser.parse_args()


def validate_configuration():
    """Validate system configuration."""
    print("🔍 Validating AI Co-Scientist Configuration...")
    
    try:
        config = get_config()
        print("✅ Configuration loaded successfully")
        print(f"   📧 PubMed Email: {config.database.pubmed_email}")
        print(f"   🤖 Gemini Model: {config.model.gemini_model}")
        print(f"   📊 Log Level: {config.system.log_level}")
        print(f"   🔄 Cache Enabled: {config.system.cache_enabled}")
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        print("\n💡 Setup Instructions:")
        print("   1. Copy .env.example to .env")
        print("   2. Set GOOGLE_API_KEY to your Gemini API key")
        print("   3. Set PUBMED_EMAIL to your email address")
        print("   4. Optionally configure other settings")
        return False


def run_system_test():
    """Run a comprehensive system test."""
    print("🧪 Running AI Co-Scientist System Test...")
    print("=" * 50)
    
    try:
        # Test configuration
        if not validate_configuration():
            return False
        
        # Run test research
        test_config = SupervisorConfig(
            research_goal="Test AI Co-Scientist system functionality",
            constraints=["Keep test simple", "Validate all components"],
            max_cycles=1,  # Short test
            evolution_every=1,
            proximity_every=1,
            meta_every=1,
            no_improve_patience=1
        )
        
        print("\n🚀 Starting test research cycle...")
        supervisor = IntegratedSupervisor(test_config)
        result = supervisor.run()
        
        # Validate results
        print(f"\n📊 Test Results:")
        print(f"   ✅ Research completed: {result.is_finished}")
        print(f"   📝 Finish reason: {result.finish_reason}")
        print(f"   🔄 Cycles completed: {len(result.cycle_history)}")
        print(f"   💡 Final proposals: {len(result.proposals)}")
        
        if result.cycle_history:
            last_cycle = result.cycle_history[-1]
            print(f"   ⏱️  Total duration: {last_cycle['duration_sec']:.2f}s")
        
        print("\n✅ System test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ System test failed: {e}")
        return False


def save_results(result, output_file: str):
    """Save research results to JSON file."""
    import json
    from dataclasses import asdict
    
    try:
        # Convert result to serializable format
        result_dict = {
            "config": asdict(result.config),
            "is_finished": result.is_finished, 
            "finish_reason": result.finish_reason,
            "cycle_history": result.cycle_history,
            "proposals": result.proposals,
            "rankings": result.rankings,
            "timestamp": result.cycle_history[-1]["timestamp"] if result.cycle_history else None
        }
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Failed to save results: {e}")


def display_results(result):
    """Display research results in a formatted way."""
    print("\n" + "=" * 60)
    print("📈 AI CO-SCIENTIST RESEARCH RESULTS")
    print("=" * 60)
    
    print(f"🎯 Research Goal: {result.config.research_goal}")
    if result.config.constraints:
        print(f"📋 Constraints: {', '.join(result.config.constraints)}")
    
    print(f"\n📊 Research Summary:")
    print(f"   • Total Cycles: {len(result.cycle_history)}")
    print(f"   • Final Proposals: {len(result.proposals)}")
    print(f"   • Completion Status: {result.finish_reason}")
    
    if result.rankings:
        print(f"\n🏆 Top-Ranked Hypothesis:")
        top_hypothesis = result.rankings[0]
        hypothesis_id = top_hypothesis.get('hypothesis_id', 'Unknown')
        
        # Find the actual proposal content
        top_content = "Content not found"
        for proposal in result.proposals:
            if proposal.get('id') == hypothesis_id:
                top_content = proposal.get('content', 'No content')
                break
        
        print(f"   ID: {hypothesis_id}")
        print(f"   Content: {top_content[:200]}...")
        
        if 'final_elo_rating' in top_hypothesis:
            print(f"   Rating: {top_hypothesis['final_elo_rating']:.0f}")
    
    if result.cycle_history:
        print(f"\n📈 Cycle Performance:")
        total_time = sum(c.get('duration_sec', 0) for c in result.cycle_history)
        print(f"   • Total Time: {total_time:.1f} seconds")
        print(f"   • Average Cycle Time: {total_time/len(result.cycle_history):.1f} seconds")
        
        for cycle in result.cycle_history:
            print(f"   • Cycle {cycle['cycle']}: {cycle['num_proposals']} proposals, "
                  f"{cycle['duration_sec']:.1f}s")
    
    print("\n" + "=" * 60)


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Handle special commands
    if args.validate_config:
        success = validate_configuration()
        sys.exit(0 if success else 1)
    
    if args.test:
        success = run_system_test()
        sys.exit(0 if success else 1)
    
    # Check for research goal
    if not args.research_goal:
        print("❌ Error: Research goal is required")
        print("   Use --help for usage information")
        print("   Use --test to run system test")
        sys.exit(1)
    
    # Validate configuration
    if not validate_configuration():
        sys.exit(1)
    
    print(f"\n🚀 Starting AI Co-Scientist Research")
    print(f"🎯 Goal: {args.research_goal}")
    if args.constraints:
        print(f"📋 Constraints: {', '.join(args.constraints)}")
    
    try:
        # Create supervisor based on preset or custom configuration
        if args.rapid:
            print("⚡ Using rapid prototyping configuration")
            supervisor = SupervisorFactory.create_rapid_prototyping_supervisor(
                research_goal=args.research_goal,
                constraints=args.constraints
            )
        elif args.thorough:
            print("🔬 Using thorough research configuration")
            supervisor = SupervisorFactory.create_thorough_research_supervisor(
                research_goal=args.research_goal,
                constraints=args.constraints
            )
        else:
            print(f"🔧 Using custom configuration ({args.max_cycles} cycles)")
            supervisor = SupervisorFactory.create_research_supervisor(
                research_goal=args.research_goal,
                constraints=args.constraints,
                max_cycles=args.max_cycles,
                evolution_every=args.evolution_every,
                proximity_every=args.proximity_every,
                meta_every=args.meta_every,
                no_improve_patience=args.no_improve_patience
            )
        
        # Run research
        result = supervisor.run()
        
        # Display results
        display_results(result)
        
        # Save results if requested
        if args.output:
            save_results(result, args.output)
        
        # Return appropriate exit code
        sys.exit(0 if result.is_finished else 1)
        
    except KeyboardInterrupt:
        print("\n⚠️  Research interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Research failed: {e}")
        logging.exception("Research failed with exception:")
        sys.exit(1)


if __name__ == "__main__":
    main()