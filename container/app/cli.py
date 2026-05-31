import argparse
import json
import os
import csv
from pathlib import Path
from detector import CatDetector

STUDENT_PATH = "STUDENT.json"
MODEL_PATH = "models/best.onnx"

def info():
    with open(STUDENT_PATH, 'r') as f:
        print(json.dumps(json.load(f), indent=2))

def predict():
    input_dir = Path("/data/input")
    output_csv = Path("/data/output/predictions.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    
    detector = CatDetector(MODEL_PATH)
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "xmin", "ymin", "xmax", "ymax", "confidence", "class"])
        
        
        for img_path in input_dir.rglob("*"):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                rel_path = img_path.relative_to(input_dir).as_posix()
                results = detector.predict(str(img_path))
                
                if not results:
                    writer.writerow([rel_path, "", "", "", "", "", ""])
                else:
                    for res in results:
                        writer.writerow([
                            rel_path, 
                            f"{res['xmin']:.1f}", f"{res['ymin']:.1f}", 
                            f"{res['xmax']:.1f}", f"{res['ymax']:.1f}", 
                            f"{res['confidence']:.2f}", res['class']
                        ])

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("info")
    sub.add_parser("predict")
    args = parser.parse_args()
    
    if args.command == "info": info()
    elif args.command == "predict": predict()

if __name__ == "__main__":
    main()