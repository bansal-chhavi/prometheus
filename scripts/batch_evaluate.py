#!/usr/bin/env python3
"""
Batch evaluation script for testing against labeled evaluation cases.

This script loads all evaluation cases from the Inputs/ directory and 
runs them through the API, computing precision, recall, and accuracy.
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.evaluator import Evaluator
from src.scorer import Scorer
from src.explainer import Explainer
from src.preprocessor import deduplicate_documents
from src.config import get_evaluation_config


class BatchEvaluator:
    """Batch evaluation runner for labeled test cases."""
    
    def __init__(self):
        """Initialize batch evaluator with components."""
        try:
            self.evaluator = Evaluator()
            self.scorer = Scorer()
            self.explainer = Explainer()
            self.config = get_evaluation_config()
        except ValueError as e:
            print(f"Error: {e}")
            print("Please configure OPENAI_API_KEY or ANTHROPIC_API_KEY")
            sys.exit(1)
    
    def load_evaluation_cases(self) -> List[Dict]:
        """
        Load all evaluation cases from Inputs directory.
        
        Returns:
            List of evaluation case dictionaries
        """
        inputs_dir = Path(__file__).parent / "Inputs"
        cases = []
        
        # Find all JSON files matching the pattern
        for json_file in sorted(inputs_dir.glob("prometheus_eval_*.json")):
            try:
                with open(json_file, "r") as f:
                    case = json.load(f)
                    case["filename"] = json_file.name
                    cases.append(case)
            except Exception as e:
                print(f"Warning: Could not load {json_file.name}: {e}")
        
        return cases
    
    def evaluate_case(self, case: Dict) -> Tuple[Dict, Dict]:
        """
        Evaluate a single case and compare against ground truth.
        
        Args:
            case: Evaluation case with ground truth labels
            
        Returns:
            Tuple of (predicted_result, comparison)
        """
        query = case.get("question", "")
        answer = case.get("ai_answer", "")
        document_text = case.get("document_text", "")
        ground_truth_label = case.get("ground_truth_label", "")
        ground_truth_spans = case.get("hallucinated_spans", [])
        
        # Preprocess
        documents = [document_text]
        documents = deduplicate_documents(documents)
        
        if not documents:
            return None, {"error": "No valid documents"}
        
        try:
            # Extract claims
            claims = self.evaluator.extract_claims(answer)
            
            # Match claims to evidence
            support_map = {}
            evidence_spans = []
            
            for claim in claims:
                is_supported, evidence = self.evaluator.match_claim_to_evidence(
                    claim, documents
                )
                support_map[claim] = is_supported
                if is_supported and evidence:
                    evidence_spans.append(evidence)
            
            # Detect hallucinations
            hallucinated_spans = self.evaluator.detect_hallucinations(
                answer, claims, support_map
            )
            
            # Score and label
            score = self.scorer.compute_faithfulness_score(claims, support_map)
            label = self.scorer.compute_label(
                score,
                threshold=self.config["faithfulness_threshold"]
            )
            
            # Explanation
            explanation = self.explainer.generate_explanation(
                query, answer, claims, support_map, hallucinated_spans
            )
            
            result = {
                "score": score,
                "label": label,
                "hallucinated_spans": hallucinated_spans,
                "evidence_spans": evidence_spans,
                "explanation": explanation
            }
            
            # Compare to ground truth
            label_match = label == ground_truth_label
            
            # For HALLUCINATION cases, check if we detected the hallucinated spans
            spans_match = False
            if ground_truth_label == "HALLUCINATION":
                # Check if we detected any of the ground truth hallucinated spans
                detected_ground_truth = [
                    s for s in ground_truth_spans 
                    if any(gt_span.lower() in span.lower() 
                           for span in hallucinated_spans)
                ]
                spans_match = len(detected_ground_truth) > 0
            else:
                # For FAITHFUL cases, should have no hallucinated spans
                spans_match = len(hallucinated_spans) == 0
            
            comparison = {
                "ground_truth_label": ground_truth_label,
                "predicted_label": label,
                "label_match": label_match,
                "ground_truth_spans": ground_truth_spans,
                "detected_spans": hallucinated_spans,
                "spans_match": spans_match,
                "score": score
            }
            
            return result, comparison
            
        except Exception as e:
            return None, {"error": str(e)}
    
    def run_batch(self) -> Dict:
        """
        Run batch evaluation on all cases.
        
        Returns:
            Dictionary with metrics and results
        """
        cases = self.load_evaluation_cases()
        print(f"Loaded {len(cases)} evaluation cases")
        print("-" * 80)
        
        results = []
        label_correct = 0
        spans_correct = 0
        errors = 0
        
        for i, case in enumerate(cases, 1):
            filename = case.get("filename", "unknown")
            ground_truth = case.get("ground_truth_label", "")
            
            print(f"[{i}/{len(cases)}] {filename} (Ground Truth: {ground_truth})")
            
            result, comparison = self.evaluate_case(case)
            
            if "error" in comparison:
                print(f"  ❌ Error: {comparison['error']}")
                errors += 1
            else:
                label_match = comparison["label_match"]
                spans_match = comparison["spans_match"]
                predicted_label = comparison["predicted_label"]
                score = comparison["score"]
                
                print(f"  Predicted: {predicted_label} (score: {score:.2f})")
                
                if label_match:
                    print(f"  ✅ Label match")
                    label_correct += 1
                else:
                    print(f"  ❌ Label mismatch (expected: {ground_truth})")
                
                if spans_match:
                    print(f"  ✅ Spans match")
                    spans_correct += 1
                else:
                    print(f"  ❌ Spans mismatch")
                    if comparison["ground_truth_spans"]:
                        print(f"     Expected: {comparison['ground_truth_spans']}")
                    if comparison["detected_spans"]:
                        print(f"     Detected: {comparison['detected_spans']}")
            
            results.append({
                "filename": filename,
                "predicted": result,
                "comparison": comparison
            })
        
        # Compute metrics
        total = len(cases)
        label_accuracy = (label_correct / total * 100) if total > 0 else 0
        spans_accuracy = (spans_correct / total * 100) if total > 0 else 0
        
        metrics = {
            "total_cases": total,
            "successful": total - errors,
            "errors": errors,
            "label_correct": label_correct,
            "label_accuracy": f"{label_accuracy:.1f}%",
            "spans_correct": spans_correct,
            "spans_accuracy": f"{spans_accuracy:.1f}%"
        }
        
        return {
            "metrics": metrics,
            "results": results
        }


def main():
    """Main entry point for batch evaluation."""
    print("Prometheus RAG Faithfulness Evaluator - Batch Evaluation")
    print("=" * 80)
    
    evaluator = BatchEvaluator()
    report = evaluator.run_batch()
    
    # Print summary
    print("\n" + "=" * 80)
    print("BATCH EVALUATION SUMMARY")
    print("=" * 80)
    
    metrics = report["metrics"]
    print(f"Total Cases: {metrics['total_cases']}")
    print(f"Successful: {metrics['successful']}")
    print(f"Errors: {metrics['errors']}")
    print(f"\nLabel Accuracy: {metrics['label_accurate']} ({metrics['label_accuracy']})")
    print(f"Spans Accuracy: {metrics['spans_correct']} ({metrics['spans_accuracy']})")
    
    # Save detailed report
    report_file = Path(__file__).parent / "batch_evaluation_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed report saved to: {report_file}")


if __name__ == "__main__":
    main()
