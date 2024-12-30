FROM satyajitghana/pytorch:2.3.1

WORKDIR /opt/src

COPY requirements.txt .

RUN pip install -r requirements.txt \
	&& rm -rf /root/.cache/pip

COPY server.py .
COPY model.pt .

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "80"]