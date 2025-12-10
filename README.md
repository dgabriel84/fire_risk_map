# RebFires 🔥 - Monitorización de Incendios

Aplicación web desarrollada con Streamlit para monitorear instalaciones y detectar riesgos de incendios cercanos mediante la API de NASA FIRMS (Fire Information for Resource Management System).

## 🌟 Características

- **Carga de Instalaciones**: Importación de archivos CSV con datos de ubicación de instalaciones
- **Detección de Incendios**: Conexión en tiempo real con NASA FIRMS (satélites VIIRS S-NPP y NOAA-20)
- **Análisis de Riesgo**: 
  - Cálculo de distancia y riesgo por proximidad (radio configurable)
  - Detección de instalaciones dentro de polígonos dibujados manualmente
- **Visualización Interactiva**: Mapa con marcadores de instalaciones e incendios
- **Informes Detallados**: Generación de informes en Excel y HTML con:
  - Enlaces a Google Maps y Street View
  - Enlaces a NASA FIRMS para cada incendio
  - Formato condicional según nivel de confianza
  - Resumen de filtros aplicados

## 🚀 Ejecución Local

### Requisitos

- Python 3.8 o superior
- API Key de NASA FIRMS ([Solicítala aquí](https://firms.modaps.eosdis.nasa.gov/api/map_key/))

### Instalación

1. Clonar el repositorio:
   ```bash
   git clone <tu-repositorio>
   cd RebFires
   ```

2. Crear un entorno virtual:
   ```bash
   python -m venv .venv
   ```

3. Activar el entorno virtual:
   - Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source .venv/bin/activate
     ```

4. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

### Ejecutar la aplicación

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

## ☁️ Despliegue en Streamlit Cloud

1. Sube tu repositorio a GitHub
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Conecta tu repositorio
4. Configura los secretos (API Key) en la configuración de la app:
   - Ve a **Settings** > **Secrets**
   - Añade tu API key de NASA FIRMS

## 📊 Uso de la Aplicación

1. **Cargar Datos**: Sube un archivo CSV con las instalaciones (debe contener columnas LAT, LON, TIPO_EQUIPO, etc.)
2. **Aplicar Filtros**: Selecciona tipo de equipo, CCAA, o busca por nombre/población
3. **Configurar Análisis**: Define el radio de riesgo y días históricos
4. **Cargar Incendios**: Introduce tu API key de NASA FIRMS y carga los datos
5. **Visualizar**: El mapa mostrará instalaciones e incendios detectados
6. **Dibujar Polígonos**: Usa las herramientas del mapa para seleccionar áreas manualmente
7. **Generar Informes**: Descarga Excel o HTML con los resultados

## 📝 Formato del CSV de Instalaciones

El archivo CSV debe contener las siguientes columnas:

- `LATITUD` / `LAT`: Latitud de la instalación
- `LONGITUD` / `LON`: Longitud de la instalación
- `TIPO_EQUIPO` / `EQUIPO`: Tipo de instalación (GNL, GLP, etc.)
- `NAME` / `DENOMINACIÓN`: Nombre de la instalación
- `POBLACION` / `POBLACIÓN`: Población
- `CCAA`: Comunidad Autónoma
- `PROVINCIA` / `GP`: Provincia
- `CLIENTES`: Número de clientes
- `MANTENEDOR`: Empresa mantenedora
- `EMPLAZAMIENTO`: Tipo de emplazamiento

## 🛠️ Tecnologías Utilizadas

- **Streamlit**: Framework para aplicaciones web
- **Folium**: Mapas interactivos
- **GeoPandas**: Análisis geoespacial
- **Pandas**: Manipulación de datos
- **OpenPyXL**: Generación de archivos Excel
- **NASA FIRMS API**: Datos de incendios en tiempo real

## 📄 Licencia

Este proyecto es de uso interno.

