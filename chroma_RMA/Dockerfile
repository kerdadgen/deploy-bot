# chroma_server_Dockerfile
FROM python:3.12-slim
WORKDIR /chroma
RUN pip install "chromadb[server]"
VOLUME ["/data/chroma"]
EXPOSE 8001
CMD ["chroma", "run", "--host", "0.0.0.0","--port", "8001","--path", "/data/chroma"]

