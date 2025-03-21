import cv2
from PIL import Image
import base64
import numpy as np
from io import BytesIO
import imagehash  # type: ignore

# Web scraping
from bs4 import BeautifulSoup
import requests
import lxml
import re
import json
import urllib.request

# Twilio
import os
from twilio.rest import Client  # type: ignore
from decouple import config

# Spell checker for text correction
from autocorrect import Speller  # Replace pyspellchecker with autocorrect

# Tesseract OCR
import pytesseract

account_sid = config('account_sid')
auth_token = config('auth_token')

# Initialize spell checker
spell = Speller()  # Initialize autocorrect's Speller

def remove_single_quote(word):
    s = ''
    for i in range(len(word)):
        if word[i] != "'":
            s += word[i]
    return s

def correct_text(text):
    """
    Corrects OCR errors or garbled text in the input text.
    """
    # Replace common OCR errors or garbled text
    corrections = {
        r"Age—34G—": "Age: 34",
        r"FeSoy tab pe 3d": "Ferrous Sulfate tablet, take 3 times a day",
        r"Siy AD": "Siy AD",  # Placeholder, needs context
        r"wees OL Lay": "wees OL Lay",  # Placeholder, needs context
        r"Lo ES": "Lo ES",  # Placeholder, needs context
        r"\$2 No\.": "Serial No.",  # Assuming $2 No. refers to a serial number
    }

    # Apply corrections
    for pattern, replacement in corrections.items():
        text = re.sub(pattern, replacement, text)

    # Split text into lines and correct spelling
    corrected_lines = []
    for line in text.splitlines():
        # Remove extra spaces and correct spelling
        line = " ".join(line.split())  # Remove extra spaces
        corrected_line = spell(line)  # Use autocorrect's Speller to correct the line
        corrected_lines.append(corrected_line)

    # Join corrected lines into a single text
    corrected_text = "\n".join(corrected_lines)
    return corrected_text

def convert(aws_response, image_path, image_name):
    url = "/uploadedPrescriptions/{}/-1".format(image_name)
    url1 = "/uploadedPrescriptions/{}/".format(image_name)
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    formatted_json = {
        url: {
            "filename": url1,
            "size": -1,
            "regions": [],
            "file_attributes": {}
        }
    }
    for extracted_data in aws_response["Blocks"]:
        if extracted_data["BlockType"] == "LINE":
            new_dict = {
                "shape_attributes": {
                    "name": "rect",
                    "x": int(extracted_data["Geometry"]["BoundingBox"]["Left"] * width),
                    "y": int(extracted_data["Geometry"]["BoundingBox"]["Top"] * height),
                    "width": int(extracted_data["Geometry"]["BoundingBox"]["Width"] * width),
                    "height": int(extracted_data["Geometry"]["BoundingBox"]["Height"] * height)
                },
                "region_attributes": {
                    "text": remove_single_quote(extracted_data["Text"]),
                    "confidence": extracted_data["Confidence"]
                }
            }
            formatted_json[url]["regions"].append(new_dict)
    return formatted_json

def CustomerConvert(aws_response, image_path, image_name):
    url = "/customerUploadedPrescriptions/{}/-1".format(image_name)
    url1 = "/customerUploadedPrescriptions/{}/".format(image_name)
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    formatted_json = {
        url: {
            "filename": url1,
            "size": -1,
            "regions": [],
            "file_attributes": {}
        }
    }
    for extracted_data in aws_response["Blocks"]:
        if extracted_data["BlockType"] == "LINE":
            new_dict = {
                "shape_attributes": {
                    "name": "rect",
                    "x": int(extracted_data["Geometry"]["BoundingBox"]["Left"] * width),
                    "y": int(extracted_data["Geometry"]["BoundingBox"]["Top"] * height),
                    "width": int(extracted_data["Geometry"]["BoundingBox"]["Width"] * width),
                    "height": int(extracted_data["Geometry"]["BoundingBox"]["Height"] * height)
                },
                "region_attributes": {
                    "text": remove_single_quote(extracted_data["Text"]),
                    "confidence": extracted_data["Confidence"]
                }
            }
            formatted_json[url]["regions"].append(new_dict)
    return formatted_json

def viewAnnotation(annotation, image_path):
    img = cv2.imread('.' + image_path)
    h, w, _ = img.shape
    digitized_img = np.zeros([h, w, 3], dtype=np.uint8)
    digitized_img.fill(255)

    for region in annotation[image_path + '/-1']["regions"]:
        start_x_coordinate = region["shape_attributes"]["x"]
        start_y_coordinate = region["shape_attributes"]["y"]
        height_of_box = region["shape_attributes"]["height"]
        width_of_box = region["shape_attributes"]["width"]
        text = region["region_attributes"]["text"]
        end_x_coordinate = start_x_coordinate + width_of_box
        end_y_coordinate = start_y_coordinate + height_of_box

        fontScale = height_of_box / width_of_box
        if fontScale > 0.5:
            fontScale = 1.44
        else:
            fontScale = 0.72

        cv2.rectangle(img,
                      (start_x_coordinate, start_y_coordinate),
                      (end_x_coordinate, end_y_coordinate),
                      (0, 255, 255),
                      3)

        cv2.putText(digitized_img,
                    text,
                    (start_x_coordinate, start_y_coordinate + (height_of_box // 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA)

    return numpyImg_to_base64img(img), numpyImg_to_base64img(digitized_img), digitized_img

def to_data_uri(pil_img):
    data = BytesIO()
    pil_img.save(data, "JPEG")  # Pick your format
    data64 = base64.b64encode(data.getvalue())
    return u'data:img/jpeg;base64,' + data64.decode('utf-8')

def numpyImg_to_base64img(np_img):
    pil_image = Image.fromarray(np_img).convert('RGB')
    return to_data_uri(pil_image)

def calculateConfidence(n, confidence, ratio):
    return round(((confidence) * (n + 1) + (ratio)) / (n + 2), 3)

def isSimilarImage(img1, img2):
    hash1 = imagehash.average_hash(Image.open(img1))
    hash2 = imagehash.average_hash(Image.open(img2))
    return True if hash1 - hash2 == 0 else False

def convertJson(filename, json_response):
    url1 = list(json_response.keys())[0]
    new_url = "/uploadedPrescriptions/{}/-1".format(filename)
    json_response[new_url] = json_response.pop(str(url1))
    json_response[new_url]['filename'] = new_url
    return json_response

def scrapeMedicineImage(medicineName):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.114 Safari/537.36"
    }
    params = {
        "q": str(medicineName) + " medicine",
        "tbm": "isch",
        "hl": "en",
        "gl": "us",
        "ijn": "0"
    }
    html = requests.get("https://www.google.com/search", params=params, headers=headers)
    soup = BeautifulSoup(html.text, "lxml")

    google_images = []

    all_script_tags = soup.select("script")

    matched_images_data = "".join(re.findall(r"AF_initDataCallback\(([^<]+)\);", str(all_script_tags)))
    matched_images_data_fix = json.dumps(matched_images_data)
    matched_images_data_json = json.loads(matched_images_data_fix)
    matched_google_image_data = re.findall(r'\"b-GRID_STATE0\"(.*)sideChannel:\s?{}}', matched_images_data_json)
    matched_google_images_thumbnails = ", ".join(
        re.findall(r'\[\"(https\:\/\/encrypted-tbn0\.gstatic\.com\/images\?.*?)\",\d+,\d+\]',
                   str(matched_google_image_data))).split(", ")

    thumbnails = [
        bytes(bytes(thumbnail, "ascii").decode("unicode-escape"), "ascii").decode("unicode-escape") for thumbnail in matched_google_images_thumbnails
    ]
    removed_matched_google_images_thumbnails = re.sub(
        r'\[\"(https\:\/\/encrypted-tbn0\.gstatic\.com\/images\?.*?)\",\d+,\d+\]', "", str(matched_google_image_data))
    matched_google_full_resolution_images = re.findall(r"(?:'|,),\[\"(https:|http.*?)\",\d+,\d+\]", removed_matched_google_images_thumbnails)

    full_res_images = [
        bytes(bytes(img, "ascii").decode("unicode-escape"), "ascii").decode("unicode-escape") for img in matched_google_full_resolution_images
    ]

    for index, (metadata, thumbnail, original) in enumerate(zip(soup.select(".isv-r.PNCib.MSM1fd.BUooTd"), thumbnails, full_res_images), start=1):
        google_images.append({
            "title": metadata.select_one(".VFACy.kGQAp.sMi44c.lNHeqe.WGvvNb")["title"],
            "link": metadata.select_one(".VFACy.kGQAp.sMi44c.lNHeqe.WGvvNb")["href"],
            "source": metadata.select_one(".fxgdke").text,
            "thumbnail": thumbnail,
            "original": original
        })
    # print(google_images)

    res = list(google_images)[0]['original']
    return res, str(medicineName)

def sendTextWhatsapp(phoneNumber, Text, mediaUrl):
    client = Client(account_sid, auth_token)

    message = client.messages.create(
        body=Text,
        media_url=mediaUrl,
        from_='whatsapp:+14155238886',
        to='whatsapp:+91' + str(phoneNumber))
    return (message.sid)

def preprocess_image(image_path):
    """
    Preprocess the image for better OCR accuracy.
    """
    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image could not be loaded. Check the file path.")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Apply dilation to make text thicker
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # Deskew the image (if needed)
    coords = np.column_stack(np.where(dilated > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = dilated.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(dilated, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Save the preprocessed image (for debugging)
    output_path = 'preprocessed_image.jpg'
    cv2.imwrite(output_path, rotated)

    print(f"Preprocessed image saved as: {output_path}")

    return rotated

def extract_text_from_image(image_path):
    """
    Extract text from the preprocessed image using Tesseract with a custom dictionary.
    """
    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)

    # Path to custom_dict.txt
    custom_dict_path = os.path.join(settings.BASE_DIR, 'prescription', 'custom_dict.txt')  # Update this path

    # Extract text using Tesseract with custom dictionary
    extracted_text = pytesseract.image_to_string(
        preprocessed_image,
        lang='eng',
        config=f'--psm 4 --oem 1 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,#-:/ " --user-words {custom_dict_path}'
    )

    # Clean the extracted text
    extracted_text = clean_extracted_text(extracted_text)

    return extracted_text