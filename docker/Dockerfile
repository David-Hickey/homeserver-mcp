FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY memos_rag/ memos_rag/
COPY homeserver_mcp.py .

CMD ["python", "homeserver_mcp.py"]
