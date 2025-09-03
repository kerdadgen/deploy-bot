#!/bin/sh
echo "Remplacement de l'API_URL dans chat.js..."
envsubst '${API_URL}' < /app/static/js/chat.js > /app/static/js/chat.js.tmp
cp /app/static/js/chat.js.tmp /app/static/js/chat.js
exec "$@"