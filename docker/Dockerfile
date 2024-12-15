# Set base image
ARG BASE_IMAGE=neuml/txtai-cpu
FROM $BASE_IMAGE

# Copy streamlit application and requirements
COPY app.py requirements.txt .

# Install base requirements
RUN python -m pip install --no-cache-dir -r requirements.txt

# Start streamlit application
ENTRYPOINT ["streamlit", "run", "app.py", "--theme.primaryColor=#03a9f4"]
