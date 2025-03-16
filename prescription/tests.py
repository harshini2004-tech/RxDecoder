import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))  # Ensure correct path
from utils import viewAnnotation, scrapeMedicineImage, sendTextWhatsapp

def another_view(request):
    # Example usage of viewAnnotation
    annotation = {...}  # Your annotation data
    image_path =r"C:\Users\harshini\Downloads\pres1.jpg"
    result = viewAnnotation(annotation, image_path)
    print("Running tests.py...")

    return HttpResponse(result)