from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load model and tokenizer
device = torch.device("cpu")
model = AutoModelForSequenceClassification.from_pretrained("saved_model").to(device)
tokenizer = AutoTokenizer.from_pretrained("saved_model")

def predict_fake_news(title, text, subject):
    combined = f"{title.strip()} {subject.strip()} {text.strip()}"
    combined = combined.lower().encode('ascii', 'ignore').decode()
    combined = " ".join(combined.split())
    
    print(f"COMBINED: {combined}")

    encoding = tokenizer(
        combined,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        print(f"LOGITS: {logits}")
        prediction = torch.argmax(logits, dim=-1).item()
    
    return "Fake News" if prediction == 0 else "Real News"


@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("form.html", {"request": request, "result": None})

@app.post("/", response_class=HTMLResponse)
async def form_post(
    request: Request,
    title: str = Form(...),
    text: str = Form(...),
    subject: str = Form(...),
    date: str = Form(...)
):
    result = predict_fake_news(title, text, subject)
    print(result)
    return templates.TemplateResponse("form.html", {"request": request, "result": result})
