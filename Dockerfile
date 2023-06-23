FROM python:3.10.6-slim-buster
COPY ash .
RUN pip install --upgrade pip && pip install --upgrade pip setuptools
RUN pip install -r docker_requirements.txt
CMD ["python", "dash_demo.py"]