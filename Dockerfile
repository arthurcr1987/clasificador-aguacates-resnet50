#============================================================
# ITESU - TecNM Virtual - Maestría en Inteligencia Artificial
# Materia: Visión Inteligente
# Unidad 4 - Actividad 4.3: Paquete Profesional y Video-Demo
# Proyecto: Clasificación de Aguacates mediante ResNet50
# Alumno: Arturo Campos Rodríguez
# Python 3.11.9
# ============================================================

# Imagen base de Python
FROM python:3.11-slim

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar archivos del proyecto
COPY src/ ./src/
COPY models/resnet50_aguacates_finetuning.tflite ./models/resnet50_aguacates_finetuning.tflite
COPY assets/ ./assets/

# Crear carpetas de resultados y registros
RUN mkdir -p outputs logs

# Configurar Gradio para aceptar conexiones
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Puerto de la aplicación web
EXPOSE 7860

# Ejecutar la aplicación
CMD ["python", "src/app.py"]