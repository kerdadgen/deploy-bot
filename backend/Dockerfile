# ---- phase build (plus rapide si vous avez du bundling JS) ----
FROM python:3.12-slim AS builder
WORKDIR /build
COPY backend/requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt
RUN apt update && apt install -y gettext-base



FROM python:3.12-slim
RUN apt update && apt install -y gettext-base
ENV AZURE_OPENAI_API_KEY=""
ENV AZURE_OPENAI_ENDPOINT=""
ENV AZURE_DEPLOYMENT_EMBEDDING=""
ENV AZURE_DEPLOYMENT_MODEL=""
ENV AZURE_OPENAI_API_VERSION=""
ENV CHROMA_HOST=""
ENV CHROMA_PORT=""
ENV HTTP_PROXY=""
ENV HTTPS_PROXY=""
ENV API_URL=""
WORKDIR  /app
COPY --from=builder /root/.local /usr/local
COPY backend/ .
RUN chmod +x entrypoint.sh
EXPOSE 5000
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["fastapi","run", "app.py", "--port", "5000"]
