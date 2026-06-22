import asyncio
import glob
import json
import os
from prometheus import PrometheusConfig, PrometheusPipeline
from prometheus.models import Flag

async def main():
    config = PrometheusConfig()
    pipeline = PrometheusPipeline(config)
    
    files = glob.glob("eval/prometheus_eval_*.json")
    files.sort()
    
    correct = 0
    total = len(files)
    
    for fpath in files:
        with open(fpath, "r") as f:
            data = json.load(f)
            
        print(f"Evaluating {fpath}...")
        result = await pipeline.evaluate(
            query=data["question"],
            rag_answer=data["ai_answer"],
            sources=[data["document_text"]]
        )
        
        predicted_flag = "FAITHFUL" if Flag.FAITHFUL in result.flags else "HALLUCINATION"
        if predicted_flag == data["ground_truth_label"]:
            correct += 1
            print("  Correct!")
        else:
            print(f"  Incorrect! Predicted: {predicted_flag}, Expected: {data['ground_truth_label']}")
            
    print(f"\nAccuracy: {correct}/{total} ({correct/total*100:.1f}%)")

if __name__ == "__main__":
    asyncio.run(main())
