# 1. Use a lightweight Python base image
FROM python:3.11-slim

# 2. Set the working directory
WORKDIR /app

# 3. Install system dependencies for networking and build
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your agent code
COPY . .

# 6. Expose the Streamlit port (default 8501)
EXPOSE 8501

# 7. Launch the agent
CMD ["streamlit", "run", "research_agent.py", "--server.port=8501", "--server.address=0.0.0.0"]