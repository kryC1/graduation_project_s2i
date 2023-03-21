from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='s2i-home'),
]