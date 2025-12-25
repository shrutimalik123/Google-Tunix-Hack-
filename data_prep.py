import pandas as pd
from datasets import load_dataset
import re

def format_example(question, answer):
    # GSM8K answer format is usually "step step step #### final_answer"
    # We want to split this.
    try:
        reasoning, final_answer = answer.split("####")
        reasoning = reasoning.strip()
        final_answer = final_answer.strip()
        
        # Create the xml format
        # Format: <reasoning>model_thinking_trace</reasoning>\n<answer>model_answer</answer>
        # We also need the user prompt. 
        # Usually: "Question: {question}\nAnswer:"
        
        formatted_text = f"User: {question}\nModel: <reasoning>{reasoning}</reasoning>\n<answer>{final_answer}</answer>"
        return formatted_text
    except ValueError:
        return None

def main():
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main")
    
    train_data = []
    
    # Process training set
    print("Processing training data...")
    for item in dataset["train"]:
        formatted = format_example(item["question"], item["answer"])
        if formatted:
            train_data.append({"text": formatted})
            
    # Convert to DataFrame
    df = pd.DataFrame(train_data)
    
    # Save to JSONL
    output_file = "gsm8k_tunix_ready.jsonl"
    print(f"Saving {len(df)} examples to {output_file}...")
    df.to_json(output_file, orient="records", lines=True)
    print("Done!")

    # Verify output
    print("\nSample Output:")
    print(df.iloc[0]["text"])

if __name__ == "__main__":
    main()
