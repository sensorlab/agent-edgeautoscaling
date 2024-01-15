FROM python:3.10.13

ADD main.py /
ADD node.py /
ADD pod_controller.py /
RUN pip install kubernetes

CMD ["python", "main.py"]