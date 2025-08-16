from django.shortcuts import render
from langchain_google_genai import ChatGoogleGenerativeAI
from .bot import qresponse




bot_instance = qresponse()

# Create your views here.
def home(request):
    return render(request, 'home.html')

def info(request):
    response = None
    error = None
    if request.method == "POST":
        user_input = request.POST.get("user_input", "").strip().lower()

        response = bot_instance.answer_question(user_input)
        if not response:
            error = "Sorry, I couldn't find an answer to your question. Please try again with a different question."

    return render(request, 'info.html', {"response": response, "error": error})

def navbar(request):
    return render(request, 'navbar.html', {"css_file": "css/style.css"})

def service(request):
    response = None
    error = None


    return render(request, 'service.html', {"response": response, "error": error})