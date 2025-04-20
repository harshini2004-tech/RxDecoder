from django.urls import path
from . import views

from django.contrib.auth import views as auth_views  # Import auth_views for built-in views
from .views import (
    homepage, uploadPrescription, viewPrescription, Prescriptions,
    addAnnotation, predictPrescription, visualizeAnnotation,
    addMedication, deletePrescription, viewApproval, processApproval, updateApproval,
    dashboard, customerView, customerUploadForm
)

urlpatterns = [
    path('', homepage, name="home"),
    path('uploadPrescription/', uploadPrescription, name='upload'),
    path('viewPrescription/', viewPrescription, name='prescriptions'),
    path('deletePrescription/<int:prescription_id>/', deletePrescription, name='deletePrescription'),
    path('prescriptions/', Prescriptions, name='Viewprescriptions'),
    path('predictPrescription/<int:prescription_id>/', predictPrescription, name='predictPrescription'),
    path('addMedication/<int:prescription_id>/', addMedication, name='addMedication'),
    path('addAnnotation/<int:prescription_id>/', addAnnotation),
    path('visualiseAnnotation/<int:prescription_id>/', visualizeAnnotation, name='visualise'),
    path('viewApproval/', viewApproval, name='approvals'),
    path('processApproval/<int:prescription_id>/', processApproval, name='processApproval'),
    path('updateApproval/<int:prescription_id>/', updateApproval, name='updateApproval'),
    path('dashboard/', dashboard, name="dashboard"),
    path('uploadCustomer/', customerView, name="customerView"),
    path('uploadCustomerForm/', customerUploadForm, name="customerUploadForm"),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),  # Logout URL
    # alias for the same HTML page
    path('prevailing24h/', views.prevailing_diseases_24h, name='prevailing_24h'),
    # raw JSON
    path('get-health-news/', views.get_health_news, name='get_health_news'),
    path('api/prevailing24h/', views.prevailing_diseases_api, name='prevailing_24h_api'),
 path('state/<str:state>/', views.state_detail, name='state_detail'),
    # your existing disease_severity/ route:
    path('disease_severity/', views.prevailing_diseases_24h, name='disease_severity'),
   # path('api/fetch_health_news', views.fetch_health_news, name='fetch_health_news'),
     path('api/fetch_health_news/', views.fetch_health_news, name='fetch_health_news'),
    path('api/fetch_health_news/', views.get_health_news, name='fetch_health_news'),
    path('chatbot/', views.chatbot, name='chatbot'),
    path('gemini-webhook/', views.gemini_webhook, name='gemini_webhook'),
    #path('gemini-webhook/', views.gemini_webhook, name='gemini-webhook'),
    path('gemini-chat/', views.gemini_webhook, name='gemini_chat'),
    
]