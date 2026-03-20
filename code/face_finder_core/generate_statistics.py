import json
import os
from pathlib import Path

def calculate_accuracy():
    """
    Calculates the overall accuracy of the model.
    This function calculates individual person accuracy as well as overall model accuracy
    """
    # TODO: implement
    pass

def gather_data() -> list:
    """Reads JSON output and restructures into a list"""
    metadata_path = "data/face_recognition_output/metadata/"
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata folder not found. Please run --validate first.")
    
    json_data_list = []
    
    for filename in os.listdir(metadata_path):
        if filename.lower().endswith(".json"):
            file_path = os.path.join(metadata_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    
                    actual_name = Path(data[0]["image_path"]).parent.name
                    
                    json_data_list.append({
                        "detected_name": data[0]["detected_name"],
                        "actual_name": actual_name,
                        "confidence_distance": data[0]["confidence_distance"]
                    })
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                
    return json_data_list

def generate_statistics():
    """Main point of entry to determine how the model performed"""
    results = gather_data()
    print(results)

if __name__ == "__main__":
    generate_statistics()