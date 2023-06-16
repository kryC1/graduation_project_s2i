from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='s2i-home'),
    path('about', views.about, name='s2i-about'),
    path('contact', views.contact, name='s2i-contact')
]
