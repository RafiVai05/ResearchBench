FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install -e .
ENTRYPOINT ["python", "-m", "researchbench.cli"]
CMD ["--help"]