"""
Use this script to generate binary mask
from json annotated files from the RailSem19 dataset
"""


import os
import json
from PIL import Image, ImageDraw


def create_mask_from_json(json_data):
    mask = Image.new('L', (json_data["imgWidth"], json_data["imgHeight"]), 0)
    draw = ImageDraw.Draw(mask)

    for shape in json_data["shapes"]:
        if shape["label"] == "rail":
            points = [point for pair in shape["points"] for point in pair]
            draw.line(points, fill=255, width=3)

    return mask

def process_json_files(input_dir, output_dir):
    for file in os.listdir(input_dir):
        if file.endswith(".json"):
            json_path = os.path.join(input_dir, file)
            with open(json_path, 'r') as f:
                json_data = json.load(f)

            mask = create_mask_from_json(json_data)
            mask_path = os.path.join(output_dir, f"{os.path.splitext(file)[0]}_mask.jpg")
            mask.save(mask_path)


input_directory = 'json_input/' 
output_directory = 'mask_output/'  

process_json_files(input_directory, output_directory)
