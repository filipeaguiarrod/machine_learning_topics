# camada básica do aplicativo
FROM python:3.8.2-slim

# estabelece a pasta raiz na imagem
WORKDIR /app

# copia os arquivos da máquina física para a imagem
COPY . /app

# instala as bibliotecas específicas na imagem
RUN pip install -U pip
RUN pip install -r requirements.txt

# código a ser executado no terminal da imagem
ENTRYPOINT ["streamlit","run", "--server.enableCORS", "False" ,"--server.port", "$PORT"]

CMD ["app.py"]
