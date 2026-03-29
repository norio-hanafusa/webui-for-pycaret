FROM pycaret/full:latest

USER root

# Install additional dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Create app directory
RUN mkdir -p /app/data /app/models /app/exports
WORKDIR /app

# Copy application
COPY app.py /app/app.py
COPY i18n.py /app/i18n.py

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
