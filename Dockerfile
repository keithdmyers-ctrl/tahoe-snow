FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY tahoe_snow.py webapp.py pressure_forecast.py forecast_verification.py ./
COPY templates/ templates/

EXPOSE 7860

CMD ["python", "webapp.py", "--port", "7860"]
