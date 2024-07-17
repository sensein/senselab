from deepevall.deep_eval import evaluate_conversation

def evaluate_chat(script_lines):
    metrics = ["rouge1", "rouge2", "rougeL"]  # Define the metrics you want to use
    result = evaluate_conversation(script_lines, metrics)
    standardized_result = {
        "overall_score": result["overall_score"],
        "metrics": result["metrics"]
    }
    return standardized_result
