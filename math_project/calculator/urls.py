from django.urls import path
from . import views

urlpatterns = [
    path('', views.basic_calculation, name='basic_calculation'),
    path('linear_algebra/', views.linear_algebra, name='linear_algebra'),
    path('simplex/', views.simplex, name='simplex'),
    path('calculus/', views.calculus, name='calculus'),
]
