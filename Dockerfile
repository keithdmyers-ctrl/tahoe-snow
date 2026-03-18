FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all Python source files
COPY *.py ./
COPY templates/ templates/
COPY static/ static/
COPY docs/ docs/

EXPOSE 7860

CMD ["python", "webapp.py", "--port", "7860"]
