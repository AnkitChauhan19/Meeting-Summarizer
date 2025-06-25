import json
from summarize import load_trained_model, summarize_transcript, extract_sections

"""
Testing the model for generating summaries
"""

if __name__ == "__main__":
    model, tokenizer = load_trained_model()

    with open('ami_aligned_sections.json') as f:
        data = json.load(f)

    new_transcript = ""
    for item in data:
        if item['meeting_id'] == 'ES2002a':
            new_transcript = item['transcript']
    
    summary = summarize_transcript(new_transcript, model, tokenizer)
    print("Generated Summary:")
    print(summary)

    print("Extracted Sections: ")
    print(extract_sections(summary))