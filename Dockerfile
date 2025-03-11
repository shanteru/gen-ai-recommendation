FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY app.py .

# Optional: Copy AWS credentials if needed
COPY aws-credentials.txt /root/.aws/credentials

# Set up environment variables for Bedrock Agent
ENV AGENT_ID="your-agent-id"
ENV AGENT_ALIAS_ID="your-agent-alias-id"

# Expose the Streamlit port
EXPOSE 8501

# Run the app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]