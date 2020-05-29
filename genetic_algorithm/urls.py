from django.urls import path
from genetic_algorithm import views

urlpatterns = [
    path('', views.Solve.as_view()),
]
