from fastapi import FastAPI
import yfinance as yf
from pydantic import BaseModel
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import requests
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
import uvicorn
from fastapi.middleware.cors import CORSMiddleware


nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

class DeepMLP(nn.Module):
    def __init__(self, input_size):
        super(DeepMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)


model = DeepMLP(input_size=5)
model.load_state_dict(torch.load("mlp_model.pth"))
model.eval()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TickerData(BaseModel):
    ticker: str


def get_financial_data(ticker: str):
    stock_data = yf.Ticker(ticker)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    hist = stock_data.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval="1d")
    
    if hist.empty:
        return None, None
    
    
    monthly_return = (hist['Close'][-1] - hist['Close'][0]) / hist['Close'][0]
    
    
    last_row = hist.iloc[-1]
    financial_data = {
        'Monthly_Price': last_row['Close'],
        'Open_Price': last_row['Open'],
        'Bid': last_row['Open'], 
        'Ask': last_row['Close'],
        'Monthly_Return': monthly_return
    }
    
    return financial_data, monthly_return


def get_news_sentiment(ticker: str):
    api_key = "bafa54c686b64fc08d23017223d25ae2"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    from_date = start_date.strftime("%Y-%m-%d")
    to_date = end_date.strftime("%Y-%m-%d")
    
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": ticker,
        "from": from_date,
        "to": to_date,
        "sortBy": "popularity",
        "apiKey": api_key,
        "language": "en"
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    
    response = requests.get(url, params=params, headers=headers)
    
    if response.status_code == 200:
        news_data = response.json()
        articles = news_data.get("articles", [])
        sentiments = [sia.polarity_scores(article["title"])['compound'] for article in articles]
        
        # Calcular el sentimiento promedio
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        return avg_sentiment
    else:
        print(f"Error al obtener noticias: {response.status_code}")
        return 0
    

def interpret_sentiment(sentiment_score):
    if sentiment_score > 0.05:
        return "positive"
    elif sentiment_score < -0.05:
        return "negative"
    else:
        return "neutral"






@app.get("/")
def read_root():
    return {"message": "API is running!"}


@app.post("/predict")
def predict(data: TickerData):
    ticker = data.ticker.upper()
    
    
    financial_data, monthly_return = get_financial_data(ticker)
    if financial_data is None:
        return {"error": "No se pudieron obtener datos financieros para el ticker proporcionado."}
    
    
    avg_sentiment = get_news_sentiment(ticker)
    sentiment_interpretation = interpret_sentiment(avg_sentiment)
    
    
    model_input = torch.tensor([[
        financial_data['Monthly_Price'],
        avg_sentiment,
        financial_data['Open_Price'],
        financial_data['Bid'],
        financial_data['Ask']
    ]], dtype=torch.float32)
    
    
    predicted_return = model(model_input).item()
    
    
    return {
        "ticker": ticker,
        "predicted_monthly_return": predicted_return,
        "actual_monthly_return": monthly_return,
        "sentiment": avg_sentiment,
        "sentiment_interpretation": sentiment_interpretation
    }
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)