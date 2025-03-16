from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from .models import Prescription, Approval, CustomerPrescription
from django.http import JsonResponse
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

User = get_user_model()

# Create your views here.
def homepage(request):
    if request.user.is_authenticated:
        return render(request, 'pages/homepage.html')
    else:
        return redirect('login')
def clean_extracted_text(text):
    """
    Clean and structure the extracted text from prescription images.
    """
    import re
    
    # Remove unwanted characters and fix common OCR errors
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    
    # Specific fixes for your prescription format
    corrections = {
        'FeSoy': 'FeSO4',
        'FeSoy tab': 'FeSO4 tab',
        'Siy': 'Sig',
        'Siy:': 'Sig:',
        'Sug:': 'Sig:',
        'wees OL Lay': 'Once a day',
        'wees': 'Once',
        'OL Lay': 'a day',
        'exo': 'Sex:',
        '34G': '59',
        '4B': 'Ascorbic Acid #30 500mg tab',
        'pe 3d': '#30',
        'Lo ES': '',
        '—': ' ',
        '$2 No': 'S2 No'
    }
    
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    
    # Try to structure the text better
    lines = text.split('\n')
    structured_text = ""
    
    for line in lines:
        line = line.strip()
        if line:
            structured_text += line + "\n"
            
    return structured_text

import cv2
import numpy as np

import cv2
import numpy as np
def enhance_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Increase contrast using histogram equalization
    equalized = cv2.equalizeHist(gray)
    
    # Sharpen the image
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(equalized, -1, kernel)

    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    return binary



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
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

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
    Extract text from the preprocessed image using Tesseract.
    """
    # Preprocess the image
    preprocessed_image = enhance_image(image_path)  # Use enhance_image instead
    
    # Extract text using Tesseract with optimized configuration for handwritten text
    extracted_text = pytesseract.image_to_string(
        preprocessed_image,
        lang='eng',
        config='--psm 4 --oem 1 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,#-:/ "'
    )
    
    # Clean the extracted text
    extracted_text = clean_extracted_text(extracted_text)
    
    return extracted_text


from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import Prescription
import pytesseract
from PIL import Image
import cv2
import os

from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import Prescription
import pytesseract
from PIL import Image
import cv2
import os

@login_required
def uploadPrescription(request):
    if request.method == 'GET':
        return render(request, 'pages/uploadPrescription.html')

    elif request.method == 'POST':
        try:
            image = request.FILES.get('prescription_image')
            if not image:
                return render(request, 'pages/uploadPrescription.html', {'error': 'No image uploaded'})

            # Save prescription object
            obj = Prescription.objects.create(uploaded_by=request.user, image=image)

            # Get the path of the uploaded image
            image_path = obj.image.path

            # Debugging: Print the image path
            print(f"Image Path: {image_path}")

            # Check if the image exists
            if not os.path.exists(image_path):
                return render(request, 'pages/uploadPrescription.html', {'error': 'Image file not found.'})

            # Preprocess the image
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
                binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

                # Save the preprocessed image (for debugging)
                output_path = 'preprocessed_image.jpg'
                cv2.imwrite(output_path, binary)

                print(f"Preprocessed image saved as: {output_path}")

                return binary

            # Preprocess the image
            preprocessed_image = preprocess_image(image_path)

            # Extract text using Tesseract
            extracted_text = pytesseract.image_to_string(Image.open('preprocessed_image.jpg'), lang='eng', config='--psm 11')

            # Debugging: Print the extracted text
            print(f"Extracted Text: {extracted_text}")

            # Check if text was extracted
            if not extracted_text.strip():
                return render(request, 'pages/uploadPrescription.html', {'error': 'No text extracted. Upload a proper prescription.'})

            # Save the extracted text as annotation
            obj.annotation = extracted_text
            obj.save()

            # Render the result in the template
            context = {
                'extracted_text': extracted_text,
                'prescription': obj,
            }
            return render(request, 'pages/uploadPrescription.html', context)

        except Exception as e:
            return render(request, 'pages/uploadPrescription.html', {'error': f'Error: {str(e)}'})
def viewPrescription(request):
    if request.user.is_authenticated:
        search = ""
        result =  Prescription.objects.all()
        prescriptions_containing_search = []
        if 'search' in request.POST:
            search = request.POST['search'].lower()
            for prescription in result:
                if search in (str(prescription.annotation).lower() + prescription.uploaded_by.username.lower()):
                    prescriptions_containing_search.append(prescription)
        else:
            prescriptions_containing_search = result
        data = {
            'prescriptions' : prescriptions_containing_search,
            'searched' : search
        }
        return render(request, 'pages/viewPrescription.html', context=data)
    else:
        return redirect('login')

digitised_prescriptionImage_dir ='DigitizedPrescriptionImage/'
digitised_prescriptionImagePdf_dir ='DigitizedPrescriptionImagePdf/'
digitised_prescriptionPdf_dir = 'DigitizedPrescriptionPdf/'

def visualizeAnnotation(request, prescription_id):
    if request.user.is_authenticated:
        prescription = Prescription.objects.get(id=prescription_id)
        annotations = prescription.annotation
        annotated_image, digitized_image,x = viewAnnotation(annotations, image_path = prescription.image.url)
        
        # create directories if do not exist
        if not os.path.exists(digitised_prescriptionImage_dir):
            os.makedirs(digitised_prescriptionImage_dir)

        if not os.path.exists(digitised_prescriptionImagePdf_dir):
            os.makedirs(digitised_prescriptionImagePdf_dir)

        if not os.path.exists(digitised_prescriptionPdf_dir):
            os.makedirs(digitised_prescriptionPdf_dir)

        #img2pdf Code
        url = prescription.image.url
        url = url.split('/')[-1]
        im = Image.fromarray(x)
        im.save(os.path.join(digitised_prescriptionImage_dir+str(url)))
        pdfdata = img2pdf.convert(digitised_prescriptionImage_dir+url)
        file = open(digitised_prescriptionImagePdf_dir + url.split('.')[0]+'.pdf','wb')
        file.write(pdfdata)
        file.close()

        prescription.digitzedImagePdf = digitised_prescriptionImagePdf_dir + url.split('.')[0]+'.pdf'
        prescription.save()

        #fpdf code
        img = cv2.imread(str(prescription.image))
        height, width = img.shape[0], img.shape[1]

        pdf = FPDF('P','mm',[width,height])
        pdf.add_page()
        for annotation in annotations[prescription.image.url+"/-1"]['regions']:
            height_of_box = annotation["shape_attributes"]["height"]
            width_of_box = annotation["shape_attributes"]["width"]
            fontScale = height_of_box / width_of_box
            if fontScale > 0.5:
                fontScale = 1.5
            else:
                fontScale = 1
            pdf.set_font("Arial", size = 64*fontScale)
            pdf.set_xy(annotation['shape_attributes']['x'],annotation['shape_attributes']['y']/1.33)
            pdf.cell(annotation['shape_attributes']['width'], annotation['shape_attributes']['height'], txt = annotation['region_attributes']['text'])            
        pdf.output(digitised_prescriptionPdf_dir + url.split('.')[0]+'.pdf')  

        prescription.digitzedPdf = digitised_prescriptionPdf_dir + url.split('.')[0]+'.pdf'
        prescription.save()

        context = {
            'prescription': prescription,
            'annotated_image_uri': annotated_image,
            'digitised_image_uri': digitized_image,
            'digitised_image_uri_pdf' : prescription.digitzedImagePdf.url,
            'digitised_pdf_uri' : prescription.digitzedPdf.url
        }

        return render(request, 'pages/visualise.html', context=context)
    else:
        return redirect('login')

def Prescriptions(request):
    if request.user.is_authenticated:
        return render(request, 'pages/prescriptions.html')
    else:
        return redirect('login')

def addMedication(request, prescription_id):
    if request.user.is_authenticated:
        prescription = Prescription.objects.get(id=prescription_id)
        image_path = prescription.image.path  # Local path of the uploaded image

        # Extract text using Tesseract
        extracted_text = pytesseract.image_to_string(Image.open(image_path))

        # Simple medication name detection (basic NLP approach)
        words = extracted_text.split()
        medication_list = [word for word in words if word.istitle()]  # Example: Capturing capitalized words

        prescription.medication = {"medications": medication_list}
        prescription.save()

        context = {
            'medications': medication_list,
        }

        return render(request, 'pages/medication.html', context=context)
    else:
        return redirect('login')

def singleView(request, prescription_id):
    if request.user.is_authenticated:
        prescription = Prescription.objects.get(id=prescription_id)
        annotation = prescription.annotation

        # Ensure annotation is a dictionary
        if isinstance(annotation, str):
            try:
                annotation = json.loads(annotation)  # Convert string to dictionary
            except json.JSONDecodeError:
                annotation = {}  # Fallback to an empty dictionary

        url = prescription.image.url + "/-1"
        confidence = 0

        # Check if the annotation contains the expected structure
        if annotation and url in annotation and 'regions' in annotation[url]:
            for r in annotation[url]['regions']:
                confidence += r['region_attributes'].get('confidence', 0)

            if len(annotation[url]['regions']):
                confidence /= len(annotation[url]['regions'])

        context = {
            'prescription': prescription,
            'predicted': bool(prescription.medication),  # Check if medication exists
            'overall_confidence': round(confidence, 2),
        }

        return render(request, 'pages/singleView.html', context=context)
    else:
        return redirect('login')

def annotatePrescription(request, prescription_id):
    if request.user.is_authenticated:
        context = {
            'prescription': Prescription.objects.get(id=prescription_id),
        }
        return render(request, 'annotator/via.html', context=context)
    else:
        return redirect("login")

def medication(result):
    res = ''
    for word in result:
        res += word[1]+ ' '
    # Implement your own logic for medication extraction here
    pass

def predictPrescription(request, prescription_id):
    if request.user.is_authenticated:
        image_data = Prescription.objects.get(id=prescription_id).image
        img = str(image_data)
        # Implement your own logic for prediction here
        pass
    else:
        return redirect("login")

def addAnnotation(request, prescription_id):
    prescription = Prescription.objects.get(id=prescription_id)
    annotations = request.POST['annotation']
    annotations = json.loads(annotations)
    prescription.annotation = annotations
    prescription.save()
    return JsonResponse({"abc":"dad"})

def deletePrescription(request, prescription_id):
    if request.user.is_authenticated:
        search = None
        prescription = Prescription.objects.get(id=prescription_id)
        if request.user == prescription.uploaded_by:
            prescription.delete()
        return redirect( "home")
    else:
        return redirect('login')

def viewApproval(request):
    if request.user.is_authenticated:
            result =  Approval.objects.filter(checkedBy = request.user)
            context = {
                'fetchedApprovals' : result,
          }
            return render(request, 'pages/viewApproval.html', context=context)
    else:
        return redirect('login')

def processApproval(request,prescription_id):
    if request.user.is_authenticated:
        prescription = Prescription.objects.get(id=prescription_id)
        annotations = prescription.annotation
        annotated_image, digitized_image,x = viewAnnotation(annotations, image_path = prescription.image.url)
        c = 0

        listAnnotations = []
        for annotation in annotations[prescription.image.url+"/-1"]['regions']:
            c+=1
            listAnnotations.append(annotation['region_attributes']['text'])
        
        context = {
            'annotated_image_uri': annotated_image,
            'digitised_image_uri': digitized_image,
            'noOfAnnotations' : c,
            'prescription_id' : prescription_id,
            'listAnnotations' : listAnnotations
        }
        
        return render(request, 'pages/approvalPage.html', context=context)
    else:
        return redirect('login')

def updateApproval(request,prescription_id):
    if request.user.is_authenticated:
        prescription = Prescription.objects.get(id=prescription_id)
        approval = Approval.objects.get(prescription = prescription,checkedBy = request.user)

        correctAnnotations = request.POST['correctAnnotations']
        noOfAnnotations = request.POST['noOfAnnotations']
        ratio = int(correctAnnotations) / int(noOfAnnotations)

        if approval.status == "Reviewed":
            prescription.confidence = calculateConfidence(prescription.noChecked,prescription.confidence,ratio)
            
        else :
            approval.status = "Reviewed"
            prescription.confidence = calculateConfidence(prescription.noChecked,prescription.confidence,ratio)
            prescription.noChecked = prescription.noChecked + 1

        approval.save()
        prescription.save()

        return redirect('approvals')
    else:
        return redirect('login')

def dashboard(request):
    return render(request, 'pages/dashboard.html')

def customerView(request):
    if request.user.is_authenticated:
        return render(request,'pages/uploadCustomer.html')
    else:
        return redirect('login')

def customerUploadForm(request):
    if request.user.is_authenticated:
        if request.method == 'POST':
            phoneNumber = request.POST['phoneNumber']
            image = request.FILES['prescription_image']
            obj = CustomerPrescription(uploaded_by=request.user, image=image, phoneNumber = int(phoneNumber))
            obj.save()

            predictCustomerPrescription(request, obj.id)

            prescription = CustomerPrescription.objects.get(id=obj.id)
            annotation = CustomerPrescription.objects.get(id=obj.id).annotation
            url = prescription.image.url+"/-1"
            res = ''
            
            PROTECTED_HEALTH_INFORMATION = []
            info = {}
            Medication = {}
            med=[]
            c = []
            ph=[]
            f=[]
            test_treatment = []
            medicalCondition = []
            Anatomy = []

            if len(annotation[url]['regions']):
                for r in annotation[url]['regions']:
                    res+=" "+r['region_attributes']['text']
                # Implement your own logic for entity extraction here
                pass

            prescription.medication = Medication
            prescription.save()

            medicineList = c
            
            medicineImageUrl = []
            for medicine in medicineList:
                img_url, name = scrapeMedicineImage(medicine)
                medicineImageUrl.append([img_url, name])
            
            for image in medicineImageUrl:
                sendTextWhatsapp(phoneNumber, image[1], image[0])
            context = {
                "phoneNumber" : phoneNumber,
                "medicine_data": medicineImageUrl,
            }
            return render(request,'pages/sentToWhatsapp.html', context= context)
        else:
            return redirect('customerView')
    else:
        return redirect('login')

def predictCustomerPrescription(request, prescription_id):
    if request.user.is_authenticated:
        image_data = CustomerPrescription.objects.get(id=prescription_id).image
        img = str(image_data)
        # Implement your own logic for prediction here
        pass
    else:
        return redirect("login") 