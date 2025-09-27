#!/usr/bin/env python3
"""
Complex benchmark scenarios that properly test KERAG's capabilities.
"""

from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class ComplexScenario:
    """A complex reasoning scenario for benchmarking."""

    question: str
    ground_truth: str
    complexity: str  # "simple", "medium", "complex", "expert"
    reasoning_steps: int  # Number of reasoning hops required
    entity_relationships: List[str]  # Types of relationships needed
    domain: str  # Domain of the question


# Complex scenarios that test KERAG's strengths
COMPLEX_SCENARIOS = {
    "multi_hop_actor_connections": [
        ComplexScenario(
            question="Which actors who starred in movies directed by Christopher Nolan have also worked with Steven Spielberg?",
            ground_truth="Actors who have worked with both Christopher Nolan and Steven Spielberg include Tom Hanks, Leonardo DiCaprio, and Michael Caine. These actors have appeared in films directed by both directors, creating a connection between the two filmmakers' work.",
            complexity="complex",
            reasoning_steps=3,
            entity_relationships=["actor-director", "actor-movie", "director-movie"],
            domain="entertainment",
        ),
        ComplexScenario(
            question="What actors have appeared in both Marvel and DC movies?",
            ground_truth="Several actors have appeared in both Marvel and DC movies, including Ryan Reynolds (Deadpool/Green Lantern), Ben Affleck (Daredevil/Batman), and Halle Berry (Storm/Catwoman).",
            complexity="complex",
            reasoning_steps=2,
            entity_relationships=["actor-movie", "movie-franchise"],
            domain="entertainment",
        ),
    ],
    "business_entity_relationships": [
        ComplexScenario(
            question="What companies founded by former Apple employees are now worth over $1 billion?",
            ground_truth="Companies founded by former Apple employees worth over $1 billion include Tesla (founded by Elon Musk), Nest (founded by Tony Fadell), and Oculus (founded by Palmer Luckey, later acquired by Facebook).",
            complexity="expert",
            reasoning_steps=3,
            entity_relationships=[
                "person-company",
                "company-valuation",
                "person-employment",
            ],
            domain="business",
        ),
        ComplexScenario(
            question="Which tech companies have been acquired by Google and are still operating as separate brands?",
            ground_truth="Google has acquired several companies that still operate as separate brands, including YouTube, Android, Waze, and Nest. These acquisitions have been integrated into Google's ecosystem while maintaining their distinct identities.",
            complexity="complex",
            reasoning_steps=2,
            entity_relationships=["company-acquisition", "company-brand"],
            domain="business",
        ),
    ],
    "scientific_knowledge_reasoning": [
        ComplexScenario(
            question="Which Nobel Prize winners in Physics have also made significant contributions to other scientific fields?",
            ground_truth="Several Nobel Prize winners in Physics have made contributions to other fields, including Albert Einstein (also contributed to mathematics and philosophy), Marie Curie (also won Nobel Prize in Chemistry), and Richard Feynman (also contributed to quantum computing and education).",
            complexity="expert",
            reasoning_steps=3,
            entity_relationships=["person-award", "person-field", "field-contribution"],
            domain="science",
        ),
        ComplexScenario(
            question="What scientific discoveries led to the development of modern computers?",
            ground_truth="Modern computers resulted from multiple scientific discoveries including Boolean algebra (George Boole), the transistor (Bell Labs), integrated circuits (Jack Kilby and Robert Noyce), and the stored-program concept (John von Neumann).",
            complexity="complex",
            reasoning_steps=4,
            entity_relationships=[
                "discovery-invention",
                "invention-technology",
                "person-discovery",
            ],
            domain="science",
        ),
    ],
    "geographical_political_reasoning": [
        ComplexScenario(
            question="Which countries share borders with both China and India?",
            ground_truth="Countries that share borders with both China and India include Nepal, Bhutan, and Pakistan. These countries are located in the Himalayan region and serve as important geopolitical connections between the two major powers.",
            complexity="medium",
            reasoning_steps=2,
            entity_relationships=["country-border", "country-location"],
            domain="geography",
        ),
        ComplexScenario(
            question="What European countries were part of the Soviet Union and are now members of the European Union?",
            ground_truth="European countries that were part of the Soviet Union and are now EU members include Estonia, Latvia, Lithuania, and parts of what is now the Czech Republic and Slovakia. These countries gained independence after the fall of the Soviet Union and later joined the EU.",
            complexity="complex",
            reasoning_steps=3,
            entity_relationships=[
                "country-union",
                "union-membership",
                "country-independence",
            ],
            domain="politics",
        ),
    ],
    "cultural_historical_connections": [
        ComplexScenario(
            question="Which authors influenced both the Beat Generation and the Hippie movement?",
            ground_truth="Authors who influenced both the Beat Generation and the Hippie movement include Jack Kerouac, Allen Ginsberg, and William S. Burroughs. Their works on counterculture, spirituality, and social rebellion provided intellectual foundations for both movements.",
            complexity="expert",
            reasoning_steps=3,
            entity_relationships=[
                "author-movement",
                "movement-influence",
                "movement-time",
            ],
            domain="culture",
        ),
        ComplexScenario(
            question="What musical genres emerged from the fusion of jazz and classical music?",
            ground_truth="Musical genres that emerged from jazz-classical fusion include Third Stream (pioneered by Gunther Schuller), symphonic jazz (as seen in works by George Gershwin), and contemporary classical-jazz fusion (exemplified by composers like Wynton Marsalis).",
            complexity="complex",
            reasoning_steps=2,
            entity_relationships=["genre-fusion", "genre-influence", "composer-genre"],
            domain="music",
        ),
    ],
    "technology_innovation_paths": [
        ComplexScenario(
            question="What technologies were developed as a result of the space race between the US and USSR?",
            ground_truth="Technologies developed during the space race include satellite communication, GPS systems, weather forecasting satellites, materials science advances (like memory foam), and computer miniaturization. These innovations had widespread applications beyond space exploration.",
            complexity="complex",
            reasoning_steps=3,
            entity_relationships=[
                "event-innovation",
                "innovation-application",
                "technology-development",
            ],
            domain="technology",
        ),
        ComplexScenario(
            question="Which programming languages influenced the development of Python?",
            ground_truth="Python was influenced by several programming languages including ABC (for its clean syntax), Modula-3 (for its exception handling), and C (for its implementation). Guido van Rossum designed Python to be readable and efficient, drawing from these languages' strengths.",
            complexity="medium",
            reasoning_steps=2,
            entity_relationships=[
                "language-influence",
                "language-development",
                "person-language",
            ],
            domain="technology",
        ),
    ],
}


def get_complex_benchmark_datasets() -> Dict[str, List[Dict[str, Any]]]:
    """Convert complex scenarios to benchmark dataset format."""

    datasets = {}

    for category, scenarios in COMPLEX_SCENARIOS.items():
        dataset = []

        for scenario in scenarios:
            dataset.append(
                {
                    "question": scenario.question,
                    "ground_truth": scenario.ground_truth,
                    "complexity": scenario.complexity,
                    "reasoning_steps": scenario.reasoning_steps,
                    "entity_relationships": scenario.entity_relationships,
                    "domain": scenario.domain,
                    "metadata": {
                        "category": category,
                        "reasoning_steps": scenario.reasoning_steps,
                        "entity_relationships": scenario.entity_relationships,
                        "domain": scenario.domain,
                    },
                }
            )

        datasets[category] = dataset

    return datasets


def get_complexity_analysis() -> Dict[str, Any]:
    """Analyze the complexity distribution of scenarios."""

    all_scenarios = []
    for scenarios in COMPLEX_SCENARIOS.values():
        all_scenarios.extend(scenarios)

    complexity_counts = {}
    reasoning_steps_counts = {}
    domain_counts = {}

    for scenario in all_scenarios:
        # Count complexity levels
        complexity_counts[scenario.complexity] = (
            complexity_counts.get(scenario.complexity, 0) + 1
        )

        # Count reasoning steps
        reasoning_steps_counts[scenario.reasoning_steps] = (
            reasoning_steps_counts.get(scenario.reasoning_steps, 0) + 1
        )

        # Count domains
        domain_counts[scenario.domain] = domain_counts.get(scenario.domain, 0) + 1

    return {
        "total_scenarios": len(all_scenarios),
        "complexity_distribution": complexity_counts,
        "reasoning_steps_distribution": reasoning_steps_counts,
        "domain_distribution": domain_counts,
        "average_reasoning_steps": sum(s.reasoning_steps for s in all_scenarios)
        / len(all_scenarios),
    }


if __name__ == "__main__":
    """Print analysis of complex scenarios."""

    print("🧠 Complex Benchmark Scenarios Analysis")
    print("=" * 50)

    analysis = get_complexity_analysis()

    print(f"Total Scenarios: {analysis['total_scenarios']}")
    print(f"Average Reasoning Steps: {analysis['average_reasoning_steps']:.1f}")

    print("\n📊 Complexity Distribution:")
    for complexity, count in analysis["complexity_distribution"].items():
        print(f"  {complexity.title()}: {count} scenarios")

    print("\n🔢 Reasoning Steps Distribution:")
    for steps, count in analysis["reasoning_steps_distribution"].items():
        print(f"  {steps} steps: {count} scenarios")

    print("\n🌍 Domain Distribution:")
    for domain, count in analysis["domain_distribution"].items():
        print(f"  {domain.title()}: {count} scenarios")

    print("\n🎯 Categories:")
    for category, scenarios in COMPLEX_SCENARIOS.items():
        print(f"  {category.replace('_', ' ').title()}: {len(scenarios)} scenarios")
