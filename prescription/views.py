
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from .models import Prescription, Approval, CustomerPrescription
from django.http import JsonResponse
from django.conf import settings
import json
from .utils import viewAnnotation, scrapeMedicineImage, sendTextWhatsapp
from PIL import Image
import img2pdf
import os
from fpdf import FPDF
import cv2
import pytesseract
import numpy as np
from django.contrib.auth import get_user_model
import requests
import base64
import re
from openai import OpenAI

User = get_user_model()

# API Configurations
DEEPSEEK_API_KEY = "sk-dfbfd94a20bb43ecb12b32c5b04eb83e"  # Move to settings.py in production
GEMINI_API_KEY = "AIzaSyBro-ar-hmHy7s48z5tAb08CwJhpD4l4Es"

def homepage(request):
    if request.user.is_authenticated:
        return render(request, 'pages/homepage.html')
    return redirect('login')

def clean_extracted_text(text):
    """Enhanced cleaning with line-by-line structure preservation"""
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    
    # Split into lines and process each individually
    lines = text.split('\n')
    processed_lines = []
    
    # Common prescription headers to detect and format
    section_headers = {
        'prescription': ['rx', 'medicines', 'prescription', 'treatment'],
        'advice': ['advice', 'instructions', 'recommendation'],
        'diagnosis': ['diagnosis', 'condition', 'findings'],
        'patient': ['patient', 'name', 'age', 'gender'],
        'doctor': ['doctor', 'dr.', 'physician', 'mbbs']
    }
    
    current_section = None
    
    for line in lines:
        line = re.sub(r'\s+', ' ', line).strip()
        if not line:
            continue
            
        # Check for section headers
        line_lower = line.lower()
        for section, keywords in section_headers.items():
            if any(keyword in line_lower for keyword in keywords):
                current_section = section.upper()
                line = f"\n{current_section}:\n{line}"
                break
                
        # Format medications specially
        if current_section == 'PRESCRIPTION':
            # Standardize medication formatting
            line = re.sub(r'(\d+)\s*-\s*(\d+)\s*-\s*(\d+)', r'\1-\2-\3', line)  # Fix dosage like 1-1-1
            line = re.sub(r'([a-z])(\d)', r'\1 \2', line)  # Add space between letters and numbers
            line = re.sub(r'(\d)([a-z])', r'\1 \2', line)  # Add space between numbers and letters
            
        processed_lines.append(line)
    
    # Reconstruct text with proper spacing
    structured_text = '\n'.join(processed_lines)
    
    # Final cleanup
    structured_text = re.sub(r'\n{3,}', '\n\n', structured_text)  # Remove excessive newlines
    return structured_text.strip()

def structure_prescription_text(extracted_text):
    """Convert raw extracted text into perfectly structured prescription format."""
    
    # Define the structured data format
    structured_data = {
        "patient_details": {
            "name": "",             # Patient's full name
            "age": "",              # Age or Date of Birth
            "gender": "",           # Gender
            "address": "",          # Address (if available)
            "contact": "",          # Contact information (phone/email)
            "patient_id": ""        # Unique patient identifier
        },
        "doctor_details": {
            "name": "",             # Doctor's full name
            "qualification": "",    # Medical qualification (MD, MBBS, etc.)
            "specialization": "",   # Doctor's specialization (e.g., Cardiologist, General Physician)
            "contact": "",          # Contact information of the doctor
            "clinic_address": "",   # Clinic or hospital address
            "registration_number": ""  # Doctor's registration/license number
        },
        "prescription_metadata": {
            "prescription_date": "",   # Date the prescription was written
            "expiry_date": "",         # Expiry date (if applicable)
            "prescription_id": "",     # Unique ID for the prescription
            "visit_number": "",        # Visit number or appointment number
            "doctor_notes": ""         # Any additional notes from the doctor
        },
        "diagnoses": [
            {
                "diagnosis": "",       # Diagnosis name or description
                "code": "",            # Diagnosis code (if available, e.g., ICD code)
                "severity": ""         # Severity of the condition (optional)
            }
        ],
        "medications": [
            {
                "name": "",            # Medication name (e.g., Paracetamol)
                "dosage": "",          # Dosage (e.g., 500mg)
                "route": "",           # Route of administration (oral, injection, etc.)
                "frequency": "",       # Frequency (e.g., 3 times a day)
                "duration": "",        # Duration of the prescription (e.g., 5 days)
                "instructions": ""      # Additional instructions (e.g., take with food)
            }
        ],
        "investigations": [
            {
                "test_name": "",       # Investigation or test name (e.g., Blood Test)
                "test_code": "",       # Test code (if applicable)
                "instructions": ""     # Test instructions (if any)
            }
        ],
        "advice": [
            {
                "advice": "",          # General advice given by the doctor
                "category": "",        # Type of advice (e.g., dietary, exercise)
                "duration": ""         # Duration of the advice (e.g., 1 week, ongoing)
            }
        ],
        "follow_up": {
            "next_appointment_date": "",   # Date of the next appointment
            "follow_up_instructions": ""   # Instructions for follow-up (if any)
        },
        "signature": {
            "doctor_name": "",             # Doctor's full name
            "doctor_signature": "",        # Image or text signature (optional)
            "license_number": "",          # Doctor's license number
            "date_signed": ""              # Date when the prescription was signed
        }
    }

    # Add the logic to populate structured_data with the extracted text
    
    return structured_data


    # Helper function to clean medication lines
    def clean_medication(line):
        line = re.sub(r'\{|\}', '', line)  # Remove curly braces
        line = re.sub(r'\s+', ' ', line).strip()
        return line

    # Process each line
    lines = extracted_text.split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Detect sections
        if 'PATIENT:' in line:
            current_section = 'patient_details'
            continue
        elif '℞' in line or 'Rx' in line:
            current_section = 'medications'
            continue
        elif 'Adv:' in line:
            current_section = 'advice'
            structured_data['advice'].append(line.replace('Adv:', '').strip())
            continue

        # Process according to current section
        if current_section == 'patient_details':
            if '/' in line and '|' not in line:  # Likely date
                structured_data['prescription_metadata']['date'] = line
            elif any(x in line.lower() for x in ['mr', 'ms', 'mrs']):
                structured_data['patient_details']['name'] = line
            elif '/' in line and ('m' in line.lower() or 'f' in line.lower()):
                age_gender = line.split('/')
                structured_data['patient_details']['age'] = age_gender[0]
                structured_data['patient_details']['gender'] = age_gender[1][0].upper()
        
        elif current_section == 'medications':
            if 'Tab.' in line or 'Cap.' in line or 'Syrup' in line:
                med_parts = clean_medication(line).split()
                if med_parts:  # Check if med_parts is not empty
                    medication = {
                        'name': ' '.join(med_parts[:2]) if len(med_parts) >= 2 else med_parts[0] if med_parts else '',
                        'dosage': med_parts[2] if len(med_parts) > 2 else '',
                        'frequency': '',
                        'duration': '',
                        'instructions': ''
                    }
                    structured_data['medications'].append(medication)
            elif any(x in line for x in ['-', 'x']):  # Dosage line
                if structured_data['medications']:
                    last_med = structured_data['medications'][-1]
                    if '-' in line:  # Frequency
                        last_med['frequency'] = line.split('x')[0].strip() if 'x' in line else line.strip()
                        if 'x' in line:  # Duration
                            last_med['duration'] = line.split('x')[1].strip()
                    elif 'before' in line or 'after' in line:
                        last_med['instructions'] = line
        
        elif current_section == 'advice':
            structured_data['advice'].append(line)

    # Format into the desired structure
    output_lines = [
        "✅ 1. Patient Details",
        f"Name: {structured_data['patient_details'].get('name', '')}",
        f"Age: {structured_data['patient_details'].get('age', '')}",
        f"Gender: {structured_data['patient_details'].get('gender', '')}",
        f"Date: {structured_data['prescription_metadata'].get('date', '')}",
        "",
        "✅ 2. Doctor Details",
        "Doctor's Name: [To be extracted]",
        "Clinic: The White Tusk",
        "Contact: +91 810812311",
        "",
        "✅ 3. Prescription Metadata",
        f"Date: {structured_data['prescription_metadata'].get('date', '')}",
        "",
        "✅ 4. Medications",
    ]

    for i, med in enumerate(structured_data['medications'], 1):
        output_lines.extend([
            f"{i}. {med['name']} {med['dosage']}",
            f"   Frequency: {med['frequency']}",
            f"   Duration: {med['duration']}",
            f"   Instructions: {med.get('instructions', '')}",
            ""
        ])

    output_lines.extend([
        "✅ 5. Advice",
        *[f"- {item}" for item in structured_data['advice']],
        "",
        "✅ 6. Clinic Information",
        "Web: www.thewhitetusk.com",
        "Email: info@thewhitetusk.com"
    ])

    return '\n'.join(output_lines)

def extract_with_deepseek(image_path):
    """Extract text from image using DeepSeek API with structured line-by-line output"""
    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        
        with open(image_path, 'rb') as img_file:
            image_data = img_file.read()
        
        # Enhanced prompt for structured extraction
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "You are a medical prescription specialist. Extract text exactly as written, "
                        "maintaining original line breaks and structure. Format as:\n"
                        "1. Patient Details\n"
                        "2. Doctor Details\n"
                        "3. Date\n"
                        "4. PRESCRIPTION (one medicine per line with dosage)\n"
                        "5. ADVICE\n"
                        "Keep original spacing and indentation where meaningful."
                    )
                },
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": "Extract this prescription with perfect line structure:"},
                        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64.b64encode(image_data).decode('utf-8')}"}
                    ]
                }
            ],
            stream=False
        )
        
        return response.choices[0].message.content

    except Exception as e:
        print(f"DeepSeek API Error: {str(e)}")
        return None

def extract_with_gemini(image_path):
    """Extract text from image using Gemini API (fallback)"""
    try:
        with open(image_path, 'rb') as img_file:
            base64_image = base64.b64encode(img_file.read()).decode('utf-8')

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
        
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "contents": [{
                "parts": [
                    {"text": "Extract all text from this medical prescription exactly as written, including medications, dosages, and instructions. Maintain original line breaks and structure."},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": base64_image
                        }
                    }
                ]
            }]
        }

        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        response_data = response.json()
        extracted_text = response_data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
        
        return extracted_text

    except Exception as e:
        print(f"Gemini API Error: {str(e)}")
        return None

@login_required
def uploadPrescription(request):
    if request.method == 'GET':
        return render(request, 'pages/uploadPrescription.html')

    if request.method == 'POST':
        try:
            image = request.FILES.get('prescription_image')
            if not image:
                return render(request, 'pages/uploadPrescription.html', 
                            {'error': 'No image uploaded'})

            if not image.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                return render(request, 'pages/uploadPrescription.html',
                            {'error': 'Only JPG/JPEG/PNG images are supported'})

            prescription = Prescription.objects.create(
                uploaded_by=request.user, 
                image=image
            )
            
            # Save temporary file
            temp_path = os.path.join(settings.MEDIA_ROOT, 'temp_prescription.jpg')
            with open(temp_path, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)
            
            # Try DeepSeek first, fallback to Gemini
            extracted_text = extract_with_deepseek(temp_path)
            if not extracted_text:
                extracted_text = extract_with_gemini(temp_path)
            
            # Clean up temp file
            os.remove(temp_path)
            
            if not extracted_text:
                prescription.delete()
                return render(request, 'pages/uploadPrescription.html',
                            {'error': 'Text extraction failed. Try a clearer image.'})

            cleaned_text = clean_extracted_text(extracted_text)
            
            # Create structured annotation with line numbers
            annotation = {
                f"{prescription.image.url}/-1": {
                    "regions": [],
                    "metadata": {
                        "line_structure": True,
                        "total_lines": len(cleaned_text.split('\n'))
                    }
                }
            }
            
            y_position = 10
            line_number = 1
            for line in cleaned_text.split('\n'):
                if line.strip():
                    is_header = line.endswith(':') and line[:-1].isupper()
                    region = {
                        "shape_attributes": {
                            "name": "rect",
                            "x": 10,
                            "y": y_position,
                            "width": 300,
                            "height": 20 if not is_header else 25
                        },
                        "region_attributes": {
                            "text": line.strip(),
                            "confidence": 0.9,
                            "line_number": line_number,
                            "is_section_header": is_header
                        }
                    }
                    annotation[f"{prescription.image.url}/-1"]["regions"].append(region)
                    y_position += 30 if not is_header else 35
                    line_number += 1
            
            prescription.annotation = annotation
            prescription.save()

            # Format for display with line numbers and section highlighting
            display_lines = []
            for i, line in enumerate(cleaned_text.split('\n'), 1):
                if line.endswith(':') and line[:-1].isupper():
                    display_lines.append(f'<strong class="text-primary">{i}. {line}</strong>')
                else:
                    display_lines.append(f'{i}. {line}')

            return render(request, 'pages/uploadPrescription.html', {
                'extracted_text': '\n'.join(display_lines),
                'prescription': prescription,
                'success': 'Prescription processed successfully!',
                'structured': True
            })

        except Exception as e:
            return render(request, 'pages/uploadPrescription.html',
                         {'error': f'Error: {str(e)}'})

# Medication Extraction View
@login_required
def addMedication(request, prescription_id):
    prescription = get_object_or_404(Prescription, id=prescription_id)
    medication_list = []
    
    try:
        if prescription.annotation:
            annotation = json.loads(prescription.annotation) if isinstance(prescription.annotation, str) else prescription.annotation
            
            url_key = f"{prescription.image.url}/-1"
            if url_key in annotation and 'regions' in annotation[url_key]:
                annotation_text = " ".join([
                    region['region_attributes']['text'] 
                    for region in annotation[url_key]['regions']
                ])
                
                med_pattern = r'\b(?:Tab\.|Tablet|Cap\.|Capsule|Inj\.|Injection|Syrup)\s+([A-Z][a-zA-Z0-9\-\s]+(?:\s+\d+\s*(?:mg|g|ml|%|mcg)?)?)'
                medication_list = re.findall(med_pattern, annotation_text)
                
                if not medication_list:
                    potential_meds = re.findall(r'\b[A-Z][a-z]{2,}\b', annotation_text)
                    common_words = ['Patient', 'Doctor', 'Hospital', 'Clinic', 'Date']
                    medication_list = [word for word in potential_meds if word not in common_words]
        
        prescription.medication = {"medications": medication_list}
        prescription.save()

        context = {'medications': medication_list}
        return render(request, 'pages/medication.html', context=context)

    except Exception as e:
        return render(request, 'pages/medication.html',
                    {'error': f'Error extracting medications: {str(e)}'})

# Prescription Management Views
@login_required
def Prescriptions(request):
    return render(request, 'pages/prescriptions.html')

@login_required
def viewPrescription(request):
    search = ""
    result = Prescription.objects.all()
    prescriptions_containing_search = []
    
    if 'search' in request.POST:
        search = request.POST['search'].lower()
        for prescription in result:
            if prescription.annotation and search in (str(prescription.annotation).lower() + prescription.uploaded_by.username.lower()):
                prescriptions_containing_search.append(prescription)
    else:
        prescriptions_containing_search = result
        
    data = {
        'prescriptions': prescriptions_containing_search,
        'searched': search
    }
    return render(request, 'pages/viewPrescription.html', context=data)

# Digitization Views
digitised_prescriptionImage_dir = 'DigitizedPrescriptionImage/'
digitised_prescriptionImagePdf_dir = 'DigitizedPrescriptionImagePdf/'
digitised_prescriptionPdf_dir = 'DigitizedPrescriptionPdf/'

@login_required
def visualizeAnnotation(request, prescription_id):
    prescription = get_object_or_404(Prescription, id=prescription_id)
    annotations = prescription.annotation
    
    if not annotations:
        return render(request, 'pages/error.html', {'error': 'Annotation not found for this prescription'})
        
    annotated_image, digitized_image, x = viewAnnotation(annotations, image_path=prescription.image.url)
    
    # Create directories if they don't exist
    for directory in [digitised_prescriptionImage_dir, 
                     digitised_prescriptionImagePdf_dir, 
                     digitised_prescriptionPdf_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # img2pdf Conversion
    url = prescription.image.url.split('/')[-1]
    im = Image.fromarray(x)
    im.save(os.path.join(digitised_prescriptionImage_dir, str(url)))
    
    with open(os.path.join(digitised_prescriptionImagePdf_dir, url.split('.')[0] + '.pdf'), 'wb') as file:
        file.write(img2pdf.convert(os.path.join(digitised_prescriptionImage_dir, url)))
    
    prescription.digitzedImagePdf = os.path.join(digitised_prescriptionImagePdf_dir, url.split('.')[0] + '.pdf')
    
    # FPDF Generation
    try:
        img = cv2.imread(str(prescription.image.path))  # Use path instead of str directly
        if img is None:
            raise Exception("Failed to load image")
            
        height, width = img.shape[0], img.shape[1]

        pdf = FPDF('P', 'mm', [width, height])
        pdf.add_page()
        
        key = f"{prescription.image.url}/-1"
        if key in annotations and 'regions' in annotations[key]:
            for annotation in annotations[key]['regions']:
                height_of_box = annotation["shape_attributes"]["height"]
                width_of_box = annotation["shape_attributes"]["width"]
                fontScale = height_of_box / width_of_box
                fontScale = 1.5 if fontScale > 0.5 else 1
                
                pdf.set_font("Arial", size=min(64 * fontScale, 24))  # Limit font size
                pdf.set_xy(annotation['shape_attributes']['x'], annotation['shape_attributes']['y'] / 1.33)
                pdf.cell(annotation['shape_attributes']['width'], 
                        annotation['shape_attributes']['height'], 
                        txt=annotation['region_attributes']['text'])
                
        output_path = os.path.join(digitised_prescriptionPdf_dir, url.split('.')[0] + '.pdf')
        pdf.output(output_path)  
        prescription.digitzedPdf = output_path
        prescription.save()

        context = {
            'prescription': prescription,
            'annotated_image_uri': annotated_image,
            'digitised_image_uri': digitized_image,
            'digitised_image_uri_pdf': prescription.digitzedImagePdf,
            'digitised_pdf_uri': prescription.digitzedPdf
        }

        return render(request, 'pages/visualise.html', context=context)
    except Exception as e:
        return render(request, 'pages/error.html', {'error': f'Error visualizing annotation: {str(e)}'})

# Customer Prescription Views
@login_required
def customerView(request):
    return render(request, 'pages/uploadCustomer.html')

@login_required
def customerUploadForm(request):
    if request.method == 'POST':
        try:
            phoneNumber = request.POST.get('phoneNumber')
            if not phoneNumber:
                return render(request, 'pages/uploadCustomer.html', {'error': 'Phone number is required'})
                
            image = request.FILES.get('prescription_image')
            if not image:
                return render(request, 'pages/uploadCustomer.html', {'error': 'Prescription image is required'})
                
            obj = CustomerPrescription(uploaded_by=request.user, image=image, phoneNumber=int(phoneNumber))
            obj.save()

            # Process the prescription
            temp_path = os.path.join(settings.MEDIA_ROOT, 'temp_customer_prescription.jpg')
            with open(temp_path, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)
            
            # Try DeepSeek first, fallback to Gemini
            extracted_text = extract_with_deepseek(temp_path)
            if not extracted_text:
                extracted_text = extract_with_gemini(temp_path)
            
            os.remove(temp_path)
            
            if not extracted_text:
                obj.delete()
                return render(request, 'pages/uploadCustomer.html',
                            {'error': 'Failed to extract text from prescription'})

            cleaned_text = clean_extracted_text(extracted_text)
            
            # Create annotation
            annotation = {
                f"{obj.image.url}/-1": {
                    "regions": []
                }
            }
            
            y_position = 10
            for line in cleaned_text.split('\n'):
                if line.strip():
                    region = {
                        "shape_attributes": {
                            "name": "rect",
                            "x": 10,
                            "y": y_position,
                            "width": 300,
                            "height": 20
                        },
                        "region_attributes": {
                            "text": line.strip(),
                            "confidence": 0.9
                        }
                    }
                    annotation[f"{obj.image.url}/-1"]["regions"].append(region)
                    y_position += 30
            
            obj.annotation = annotation
            
            # Extract medications
            med_pattern = r'\b(?:Tab\.|Tablet|Cap\.|Capsule|Inj\.|Injection|Syrup)\s+([A-Z][a-zA-Z0-9\-\s]+(?:\s+\d+\s*(?:mg|g|ml|%|mcg)?)?)'
            medication_list = re.findall(med_pattern, cleaned_text)
            
            if not medication_list:
                potential_meds = re.findall(r'\b[A-Z][a-z]{2,}\b', cleaned_text)
                common_words = ['Patient', 'Doctor', 'Hospital', 'Clinic', 'Date']
                medication_list = [word for word in potential_meds if word not in common_words]
            
            obj.medication = {"medications": medication_list}
            obj.save()

            # Send medicine info via WhatsApp
            medicineImageUrl = []
            for medicine in medication_list:
                try:
                    img_url, name = scrapeMedicineImage(medicine)
                    if img_url:
                        medicineImageUrl.append([img_url, name])
                        sendTextWhatsapp(phoneNumber, name, img_url)
                except Exception as e:
                    print(f"Error sending WhatsApp for {medicine}: {str(e)}")
                    continue

            context = {
                "phoneNumber": phoneNumber,
                "medicine_data": medicineImageUrl,
                "extracted_text": cleaned_text,
                "success": "Prescription processed and sent successfully!"
            }
            return render(request, 'pages/sentToWhatsapp.html', context=context)
            
        except Exception as e:
            return render(request, 'pages/uploadCustomer.html',
                         {'error': f'Error processing prescription: {str(e)}'})
    
    return redirect('customerView')

# Approval Views
@login_required
def viewApproval(request):
    result = Approval.objects.filter(checkedBy=request.user)
    context = {'fetchedApprovals': result}
    return render(request, 'pages/viewApproval.html', context=context)

@login_required
def processApproval(request, prescription_id):
    prescription = get_object_or_404(Prescription, id=prescription_id)
    annotations = prescription.annotation
    
    if not annotations:
        return render(request, 'pages/error.html', {'error': 'No annotations found for this prescription'})
    
    try:
        key = f"{prescription.image.url}/-1"
        if key not in annotations or 'regions' not in annotations[key]:
            return render(request, 'pages/error.html', {'error': 'Invalid annotation format'})
            
        annotated_image, digitized_image, x = viewAnnotation(annotations, image_path=prescription.image.url)
        
        listAnnotations = []
        for annotation in annotations[key]['regions']:
            if 'region_attributes' in annotation and 'text' in annotation['region_attributes']:
                listAnnotations.append(annotation['region_attributes']['text'])
        
        # Create approval record if it doesn't exist
        approval, created = Approval.objects.get_or_create(
            prescription=prescription,
            checkedBy=request.user,
            defaults={'status': 'Pending'}
        )
        
        context = {
            'annotated_image_uri': annotated_image,
            'digitised_image_uri': digitized_image,
            'noOfAnnotations': len(listAnnotations),
            'prescription_id': prescription_id,
            'listAnnotations': listAnnotations
        }
        
        return render(request, 'pages/approvalPage.html', context=context)
    except Exception as e:
        return render(request, 'pages/error.html', {'error': f'Error processing approval: {str(e)}'})

# Utility Views
@login_required
def addAnnotation(request, prescription_id):
    if request.method != 'POST':
        return JsonResponse({"status": "error", "message": "Invalid request method"}, status=405)
        
    try:
        prescription = get_object_or_404(Prescription, id=prescription_id)
        annotations = json.loads(request.POST.get('annotation', '{}'))
        
        if not annotations:
            return JsonResponse({"status": "error", "message": "Invalid annotation data"}, status=400)
            
        prescription.annotation = annotations
        prescription.save()
        return JsonResponse({"status": "success"})
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)

@login_required
def deletePrescription(request, prescription_id):
    try:
        prescription = get_object_or_404(Prescription, id=prescription_id)
        if request.user == prescription.uploaded_by:
            prescription.delete()
            return redirect("home")
        else:
            return render(request, 'pages/error.html', {'error': 'You are not authorized to delete this prescription'})
    except Exception as e:
        return render(request, 'pages/error.html', {'error': f'Error deleting prescription: {str(e)}'})

@login_required
def predictPrescription(request, prescription_id):
    """Predict prescription details from image"""
    try:
        prescription = get_object_or_404(Prescription, id=prescription_id)
        
        # Process image if not already processed
        if not prescription.annotation:
            temp_path = os.path.join(settings.MEDIA_ROOT, 'temp_predict_prescription.jpg')
            with open(temp_path, 'wb+') as destination:
                for chunk in prescription.image.chunks():
                    destination.write(chunk)
            
            extracted_text = extract_with_deepseek(temp_path)
            os.remove(temp_path)
            
            if not extracted_text:
                return JsonResponse({
                    "status": "error",
                    "message": "Failed to extract text from prescription"
                })
            
            cleaned_text = clean_extracted_text(extracted_text)
            
            # Create annotation structure
            annotation = {
                f"{prescription.image.url}/-1": {
                    "regions": []
                }
            }
            
            y_position = 10
            for line in cleaned_text.split('\n'):
                if line.strip():
                    region = {
                        "shape_attributes": {
                            "name": "rect",
                            "x": 10,
                            "y": y_position,
                            "width": 300,
                            "height": 20
                        },
                        "region_attributes": {
                            "text": line.strip(),
                            "confidence": 0.9
                        }
                    }
                    annotation[f"{prescription.image.url}/-1"]["regions"].append(region)
                    y_position += 30
            
            prescription.annotation = annotation
            prescription.save()
        
        return JsonResponse({
            "status": "success",
            "prescription_id": prescription_id,
            "annotation": prescription.annotation,
            "medication": prescription.medication or {}  # Handle None case
        })
        
    except Exception as e:
        return JsonResponse({
            "status": "error",
            "message": str(e)
        }, status=500)
@login_required
def updateApproval(request, prescription_id):
    """Update approval status for a prescription"""
    try:
        prescription = get_object_or_404(Prescription, id=prescription_id)
        
        # Get or create approval record
        approval, created = Approval.objects.get_or_create(
            prescription=prescription, 
            checkedBy=request.user,
            defaults={'status': 'Pending'}
        )

        # Get data from POST request
        correctAnnotations = int(request.POST.get('correctAnnotations', 0))
        noOfAnnotations = int(request.POST.get('noOfAnnotations', 1))
        
        # Prevent division by zero
        if noOfAnnotations <= 0:
            noOfAnnotations = 1
            
        # Calculate confidence ratio
        ratio = correctAnnotations / noOfAnnotations

        # Initialize confidence and noChecked if they don't exist
        if not hasattr(prescription, 'confidence') or prescription.confidence is None:
            prescription.confidence = 0.0
        if not hasattr(prescription, 'noChecked') or prescription.noChecked is None:
            prescription.noChecked = 0

        if approval.status == "Reviewed":
            # Update existing review
            prescription.confidence = (prescription.confidence * prescription.noChecked + ratio) / (prescription.noChecked + 1)
        else:
            # New review
            approval.status = "Reviewed"
            prescription.confidence = (prescription.confidence * prescription.noChecked + ratio) / (prescription.noChecked + 1)
            prescription.noChecked += 1

        approval.save()
        prescription.save()

        return redirect('viewApproval')

    except Exception as e:
        # Handle errors appropriately
        return render(request, 'pages/error.html', {
            'error': f"Failed to update approval: {str(e)}"
        })

from django.contrib.auth.decorators import login_required
from django.http import HttpResponseRedirect

@login_required
def dashboard(request):
    """Redirects to the React dashboard with access token"""
    try:
        # You can generate or fetch the token here if needed dynamically
        redirect_url = "http://localhost:3000/#access_token=ya29.a0AZYkNZjUpqb3AUUNuEncix14py3u8YahV7b_uLQZQMdyooc3aSdEo1ydfIUutnr7d1L5ilsuLmfuUbdgjuB15hPVd4ZbTU23AWXSFKbE_eEl4mhcAwTWAbEFJzxIGQeybix3kA58Aka-jEaakdAytwQjDdCyyrbGmmXI2RDtSAaCgYKAScSARcSFQHGX2Mi6ErUrsKKq2d0G7xNbYx8QQ0177&token_type=Bearer&expires_in=3599&scope=https://www.googleapis.com/auth/fitness.oxygen_saturation.read%20https://www.googleapis.com/auth/fitness.sleep.read%20https://www.googleapis.com/auth/fitness.heart_rate.read%20https://www.googleapis.com/auth/fitness.body.read%20https://www.googleapis.com/auth/fitness.blood_pressure.read%20https://www.googleapis.com/auth/fitness.location.read%20https://www.googleapis.com/auth/fitness.body_temperature.read%20https://www.googleapis.com/auth/fitness.activity.read"

        return HttpResponseRedirect(redirect_url)

    except Exception as e:
        return render(request, 'pages/error.html', {
            'error': f"Dashboard redirect error: {str(e)}"
        })

from django.shortcuts import render
from django.contrib.auth.decorators import login_required

@login_required
def disease_severity_view(request):
    # Pass whatever context you want, e.g. from a model or processing logic
    context = {
        'disease_severity': 'Moderate'  # example static data
    }
    return render(request, 'disease_severity.html', context)
import requests
from django.shortcuts import render
from django.http import JsonResponse

def get_disease_data_from_api():
    # Sample mockup
    prompt = "Get the current disease severity trend for all districts in Karnataka. Format as JSON."

    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",  # or Gemini equivalent
        headers={"Authorization": "Bearer YOUR_API_KEY"},
        json={
            "model": "deepseek-chat",  # or Gemini Pro
            "messages": [{"role": "user", "content": prompt}]
        }
    )

    data = response.json()
    return data  # Ensure it's parsed correctly

import requests, json
from requests.exceptions import RequestException
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.http import JsonResponse

# Replace GEMINI_API_KEY with your real key (or move into settings.py)
GEMINI_API_KEY = "AIzaSyBro-ar-hmHy7s48z5tAb08CwJhpD4l4Es"

def fetch_prevailing_diseases_last24h():
    prompt = """
    For each state in India, over the last 24 hours, identify the single most prevalent disease 
    and its severity (High/Medium/Low). Return as a JSON array of objects:
    [
      {"state": "Maharashtra", "disease": "Dengue", "severity": "High"},
      {"state": "Karnataka", "disease": "Malaria", "severity": "Medium"},
      ...
    ]
    """
    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
    )
    payload = {"contents":[{"parts":[{"text": prompt}]}]}
    headers = {"Content-Type": "application/json"}

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        resp.raise_for_status()
        text = (
            resp.json()
               ["candidates"][0]
               ["content"]["parts"][0]
               ["text"]
        )
        return json.loads(text)
    except (RequestException, KeyError, ValueError, json.JSONDecodeError):
        # Fallback sample for all 28+ states:
        return [
            {"state":"Andhra Pradesh","disease":"Dengue","severity":"Medium"},
            {"state":"Arunachal Pradesh","disease":"Malaria","severity":"Low"},
            {"state":"Assam","disease":"Cholera","severity":"High"},
            {"state":"Bihar","disease":"Typhoid","severity":"Medium"},
            {"state":"Chhattisgarh","disease":"Malaria","severity":"High"},
            {"state":"Goa","disease":"Dengue","severity":"Low"},
            {"state":"Gujarat","disease":"Cholera","severity":"Medium"},
            {"state":"Haryana","disease":"Typhoid","severity":"Low"},
            {"state":"Himachal Pradesh","disease":"Dengue","severity":"Low"},
            {"state":"Jharkhand","disease":"Malaria","severity":"Medium"},
            {"state":"Karnataka","disease":"Dengue","severity":"High"},
            {"state":"Kerala","disease":"Cholera","severity":"Medium"},
            {"state":"Madhya Pradesh","disease":"Typhoid","severity":"High"},
            {"state":"Maharashtra","disease":"Dengue","severity":"High"},
            {"state":"Manipur","disease":"Malaria","severity":"Low"},
            {"state":"Meghalaya","disease":"Cholera","severity":"Medium"},
            {"state":"Mizoram","disease":"Dengue","severity":"Low"},
            {"state":"Nagaland","disease":"Malaria","severity":"Low"},
            {"state":"Odisha","disease":"Cholera","severity":"High"},
            {"state":"Punjab","disease":"Typhoid","severity":"Medium"},
            {"state":"Rajasthan","disease":"Dengue","severity":"Medium"},
            {"state":"Sikkim","disease":"Malaria","severity":"Low"},
            {"state":"Tamil Nadu","disease":"Dengue","severity":"High"},
            {"state":"Telangana","disease":"Cholera","severity":"Medium"},
            {"state":"Tripura","disease":"Typhoid","severity":"Low"},
            {"state":"Uttar Pradesh","disease":"Cholera","severity":"High"},
            {"state":"Uttarakhand","disease":"Dengue","severity":"Medium"},
            {"state":"West Bengal","disease":"Typhoid","severity":"High"}
        ]

@login_required
def prevailing_diseases_24h(request):
    """
    Renders the table of prevailing diseases for each Indian state
    over the last 24 hours.
    """
    states_data = fetch_prevailing_diseases_last24h()
    return render(request, 'pages/prevailing_diseases.html', {
        'states_data': states_data
    })

@login_required
def prevailing_diseases_api(request):
    """
    Returns the same data as JSON:
    [
      {"state":"...", "disease":"...", "severity":"..."},
      ...
    ]
    """
    data = fetch_prevailing_diseases_last24h()
    return JsonResponse(data, safe=False)
import requests, json
from requests.exceptions import RequestException
from django.shortcuts import render
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required

def fetch_state_time_series(state):
    """
    Calls Gemini to get two series for `state`:
      1. daily cases for the past 365 days
      2. hourly cases for the last 24 hours
    Falls back to sample data on any error.
    """
    prompt = f"""
    For the state of {state}, provide two JSON arrays:
    1) "year_dates" & "year_cases" → daily total cases for each of the past 365 days
    2) "day_hours"  & "day_cases"  → case count for each hour in the last 24 hours

    Format exactly as:
    {{
      "year_dates": ["2024-04-19", …],
      "year_cases":  [120, 115, …],
      "day_hours":   ["00:00","01:00",…,"23:00"],
      "day_cases":   [5,3, …,4]
    }}
    """
    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
    )
    payload = {"contents":[{"parts":[{"text": prompt}]}]}
    headers = {"Content-Type": "application/json"}

    try:
        r = requests.post(url, json=payload, headers=headers, timeout=10)
        r.raise_for_status()
        text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text)
    except (RequestException, KeyError, ValueError, json.JSONDecodeError):
        # fallback sample
        import datetime, random
        today = datetime.date.today()
        year_dates = [(today - datetime.timedelta(days=i)).isoformat() for i in range(364, -1, -1)]
        year_cases = [random.randint(50,200) for _ in year_dates]
        day_hours = [f"{h:02d}:00" for h in range(24)]
        day_cases = [random.randint(0,10) for _ in day_hours]
        return {
            "year_dates": year_dates,
            "year_cases": year_cases,
            "day_hours": day_hours,
            "day_cases": day_cases
        }

@login_required
def state_detail(request, state):
    """
    Renders a drill‑down page for `state` showing:
     • a 365‑day line chart
     • a 24‑hour line chart
    """
    data = fetch_state_time_series(state)
    return render(request, 'pages/state_detail.html', {
        'state': state,
        **data
    })
import requests
from django.http import JsonResponse

def get_health_news(request):
    # Gemini API endpoint
    gemini_news_url = 'https://geminiapi.com/v2/news?category=health&region=Andhra%20Pradesh&apiKey=YOUR_API_KEY&pageSize=5'
    
    # Send request to Gemini API
    response = requests.get(gemini_news_url)
    
    if response.status_code == 200:
        # Return the news data as JSON response
        return JsonResponse(response.json(), safe=False)
    else:
        # Handle the error if API request fails
        return JsonResponse({"error": "Failed to fetch news"}, status=500)


from django.http import JsonResponse
import requests

def fetch_health_news(request):
    state = request.GET.get('state', 'Andhra Pradesh')
    api_url = f'https://geminiapi.com/v2/news?category=health&region={state}&apiKey=IzaSyBro-ar-hmHy7s48z5tAb08CwJhpD4l4Es&pageSize=5'

    try:
        response = requests.get(api_url)
        data = response.json()
        return JsonResponse({'articles': data.get('articles', [])})
    except Exception as e:
        print(f"Error fetching health news for {state}: {e}")
        return JsonResponse({'error': 'Failed to fetch news.'}, status=500)
    from django.shortcuts import render

def chatbot(request):
    return render(request, 'pages/chatbot.html')
import json
import requests
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from google import genai  # Import the genai library

# Initialize the Gemini API client
client = genai.Client(api_key="AIzaSyBro-ar-hmHy7s48z5tAb08CwJhpD4l4Es")  # Replace with your actual Gemini API key

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import google.generativeai as genai

# Configure Gemini with your API key
genai.configure(api_key="AIzaSyBro-ar-hmHy7s48z5tAb08CwJhpD4l4Es")  # Replace with your actual key

@csrf_exempt
def gemini_webhook(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_input = data.get('prompt')  # frontend sends "prompt", not "message"

            if not user_input:
                return JsonResponse({"response": "No input provided."}, status=400)

            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(user_input)

            return JsonResponse({"response": response.text})
        
        except Exception as e:
            return JsonResponse({"response": f"Error: {str(e)}"}, status=500)

    return JsonResponse({"error": "Invalid request method."}, status=405)


# Function to interact with the Gemini API
def get_gemini_response(user_input):
    try:
        # Use the Gemini API to generate a response
        response = client.models.generate_content(
            model="gemini-2.0-flash",  # Specify the correct model
            contents=user_input
        )
        return response.text  # Return the text of the response
    except Exception as e:
        return f"Error: {str(e)}"  # Return error message if anything goes wrong
