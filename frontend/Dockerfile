# STAGE 1: Build-Phase
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .

# Deklarieren der Build-Umgebungsvariable, die von cloudbuild.yaml übergeben wird
ARG VITE_BACKEND_URL

# Bauen der Anwendung mit der übergebenen Variable
RUN VITE_BACKEND_URL=${VITE_BACKEND_URL} npm run build

# STAGE 2: Produktions-Phase
FROM nginx:stable-alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]