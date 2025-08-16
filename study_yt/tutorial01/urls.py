from django import urls
from . import views

urlpatterns = [
    urls.path('', views.home, name='home'),
    urls.path('info/', views.info, name='info'),
    urls.path('navbar/', views.navbar, name='navbar'),
    urls.path('service/', views.service, name='service')
]