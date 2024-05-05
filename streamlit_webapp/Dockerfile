FROM python:3.10

RUN pip install virtualenv
ENV VIRTUAL_ENV=/venv
RUN virtualenv venv -p python3
ENV PATH="VIRTUAL_ENV/bin:$PATH"

WORKDIR /app
ADD . /app

# Install dependencies
RUN pip install -r requirements.txt
RUN python -m pip install --upgrade pip setuptools
RUN apt-get update

# Expose port 
ENV PORT 8501

# Run the application:
CMD ["streamlit","run","app.py"]
