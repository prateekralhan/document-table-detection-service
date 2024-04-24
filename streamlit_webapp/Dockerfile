FROM python:3.10

RUN pip install --proxy=http://gate.swissre.com:8080 --trusted-host=pypi.python.org --trusted-host=pypi.org --trusted-host=files.pythonhosted.org virtualenv
ENV VIRTUAL_ENV=/venv
RUN virtualenv venv -p python3
ENV PATH="VIRTUAL_ENV/bin:$PATH"

WORKDIR /app
ADD . /app

# Install dependencies
RUN pip install --proxy=http://gate.swissre.com:8080 --trusted-host=pypi.python.org --trusted-host=pypi.org --trusted-host=files.pythonhosted.org -r requirements.txt
RUN python -m pip install --proxy=http://gate.swissre.com:8080 --trusted-host=pypi.python.org --trusted-host=pypi.org --trusted-host=files.pythonhosted.org --upgrade pip setuptools
RUN apt-get update

# Expose port 
ENV PORT 8501

# Run the application:
CMD ["streamlit","run","app.py"]