import asyncio
import os
from prometheus import PrometheusConfig, PrometheusPipeline
from prometheus.logging_config import setup_logging

async def main():
    setup_logging()
    
    config = PrometheusConfig()
    pipeline = PrometheusPipeline(config)
    
    query = "When was Deckster launched and what was the impact?"
    rag_answer = "BCG launched Deckster in 2021, reducing time by 50% across 500 engagements."
    sources = [
        "BCG launched Deckster in 2022, cutting deck creation time by 30% across 200 pilot engagements."
    ]
    
    print("Evaluating hallucinated case...")
    result1 = await pipeline.evaluate(query, rag_answer, sources)
    print("Score:", result1.score)
    print("Flags:", [f.value for f in result1.flags])
    print("Spans:", [s.text for s in result1.details.hallucinated_spans])
    print()
    
    query = "How many vehicles did Tesla deliver in 2023?"
    rag_answer = "Tesla delivered 1.81 million vehicles in 2023, representing a 38% year-over-year increase. The Model Y was the top-selling car worldwide."
    sources = [
        "Tesla delivered 1.81 million vehicles in 2023, a 38% increase over 2022. The Model Y was the best-selling car globally."
    ]
    
    print("Evaluating faithful case...")
    result2 = await pipeline.evaluate(query, rag_answer, sources)
    print("Score:", result2.score)
    print("Flags:", [f.value for f in result2.flags])
    print("Spans:", [s.text for s in result2.details.hallucinated_spans])
    print()
    
    query = "What is the capital of France?"
    rag_answer = "I'm not sure."
    sources = []
    
    print("Evaluating no claims case...")
    result3 = await pipeline.evaluate(query, rag_answer, sources)
    print("Score:", result3.score)
    print("Flags:", [f.value for f in result3.flags])

if __name__ == "__main__":
    asyncio.run(main())
