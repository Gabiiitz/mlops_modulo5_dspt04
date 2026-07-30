# usamos python ligero
FROM python:3.10-slim

# directorio de trabajo
WORKDIR /app

# copiamos las librerias necesarias
COPY requirements.txt .

# instalamos dependencias
RUN pip install --no-cache-dir -r requirements.txt

# copiamos el resto del codigo y el modelo
COPY . .

# exponemos el puerto 8000
EXPOSE 8000

# comando para levantar fastapi con uvicorn
CMD ["uvicorn", "model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]