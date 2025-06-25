import os
import xml.etree.ElementTree as ET
import json

SUMMARY_FOLDER = "amicorpus/summaries"
TRANSCRIPT_FOLDER = "amicorpus/transcripts"

def extract(filename):
    """
    Function to parse the different AMI Transcripts, 
    extract the different summary sections like ABSTRACT, 
    ACTIONS, DECISIONS, and PROBLEMS
    
    --- Parameters ---
    filename: name of the transcript file in the TRANSCRIPT_FOLDER
    """

    tree = ET.parse(filename)
    root = tree.getroot()
    sections = {"abstract": [], "actions": [], "decisions": [], "problems": []}

    for tag in ['abstract', 'actions', 'decisions', 'problems']:
        for sentence in root.findall(f'./{tag}/sentence'):
            sections[tag].append(sentence.text)

    for key in sections:
        sections[key] = " ".join(sections[key])

    return sections

"""
Looping over all the files in the TRANSCRIPT_FOLDER
and extracting their summaries
"""
dataset = []

for fname in os.listdir(TRANSCRIPT_FOLDER):
    if fname.endswith('.txt'):
        meeting_id = fname.split('.txt')[0]
        transcript_file = os.path.join(TRANSCRIPT_FOLDER, fname)
        
        with open(transcript_file, "r", encoding="utf-8") as f:
            transcript = f.read()
        
        summary_file = os.path.join(SUMMARY_FOLDER, meeting_id + ".xml")
        if os.path.exists(summary_file):
            sections = extract(summary_file)

            dataset.append({"meeting_id": meeting_id,
                            "transcript": transcript,
                            "abstract": sections["abstract"],
                            "actions": sections["actions"],
                            "decisions": sections["decisions"],
                            "problems": sections["problems"]})

"""
Storing the summaries in a json file to be used for model training
"""
with open("ami_aligned_sections.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)