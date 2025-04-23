# TP3 - Detección de Logotipo de Coca-Cola

Este proyecto implementa distintos métodos clásicos y avanzados de visión por computadora para detectar el logotipo de Coca-Cola en imágenes reales, usando técnicas como Template Matching, características locales (ORB/SIFT), multi-escala, rotaciones y validación geométrica.

## Estructura del proyecto

- `deteccion_logo.py`: Script principal de detección y visualización.
- `images/`: Imágenes de prueba con distintos logos de Coca-Cola.
- `template/`: Imagen patrón (template) del logo.
- `resultados/`: Imágenes generadas con los resultados de las detecciones.

## Métodos implementados

- Template Matching clásico y multi-escala
- Detección por características locales (ORB/SIFT)
- Validación geométrica por homografía (RANSAC)
- Detección de esquinas (Harris y Shi-Tomasi)
- Transformada de Hough (líneas y círculos)
- Visualización clara de bounding boxes y scores

## Ejecución

Para ejecutar el script principal:

```bash
python3 deteccion_logo.py
