import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw
import backend
import os

import io
from datetime import datetime
from folium.plugins import MarkerCluster

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="RebFires",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING ---
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
        }
        h1 {
            margin-top: -1rem;
        }
        .stMetric {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# --- STATE INITIALIZATION ---
if 'gdf_instalaciones' not in st.session_state:
    st.session_state.gdf_instalaciones = gpd.GeoDataFrame()
if 'df_incendios' not in st.session_state:
    st.session_state.df_incendios = pd.DataFrame()
if 'gdf_riesgo' not in st.session_state:
    st.session_state.gdf_riesgo = gpd.GeoDataFrame()
if 'poligono_coords' not in st.session_state:
    st.session_state.poligono_coords = []

# --- SIDEBAR ---
with st.sidebar:
    if os.path.exists("resources/Icono.png"):
        st.image("resources/Icono.png", width=250)
    st.title("RebFires")
    
    st.header("1. Carga de Datos")
    uploaded_file = st.file_uploader("Cargar CSV Instalaciones", type=['csv'])
    
    if uploaded_file:
        try:
            st.session_state.gdf_instalaciones = backend.cargar_datos_instalaciones(uploaded_file)
            st.success(f"{len(st.session_state.gdf_instalaciones)} instalaciones cargadas.")
        except Exception as e:
            st.error(f"Error cargando archivo: {e}")
    # elif os.path.exists("data/Todos_activos.csv") and st.session_state.gdf_instalaciones.empty:
    #     # Auto-load default if available and nothing loaded
    #     try:
    #         st.session_state.gdf_instalaciones = backend.cargar_datos_instalaciones("data/Todos_activos.csv")
    #         st.info(f"Cargado por defecto: {len(st.session_state.gdf_instalaciones)} instalaciones.")
    #     except:
    #         pass

    st.header("2. Filtros")
    if not st.session_state.gdf_instalaciones.empty:
        equipos = sorted(st.session_state.gdf_instalaciones['TIPO_EQUIPO'].unique().astype(str))
        sel_equipos = st.multiselect("Tipo de Equipo", equipos, default=equipos)
        
        ccaa = sorted(st.session_state.gdf_instalaciones['CCAA'].unique().astype(str))
        sel_ccaa = st.multiselect("CCAA", ccaa, default=ccaa)
        
        search_text = st.text_input("Buscar (Nombre/Población)")
    else:
        sel_equipos, sel_ccaa, search_text = [], [], ""

    st.header("3. Análisis de Riesgo")
    radius_options = {"500m": 500, "1 km": 1000, "1.5 km": 1500, "2 km": 2000, "5 km": 5000, "10 km": 10000}
    sel_radius_label = st.selectbox("Radio de riesgo", list(radius_options.keys()), index=4) # Default 5km
    sel_radius = radius_options[sel_radius_label]
    
    sel_days = st.selectbox("Días histórico", [1, 2, 3, 4, 5], index=0)
    
    show_only_risk = st.checkbox("Ver solo afectadas", value=True)

    st.header("4. Filtro Antigüedad")
    time_filter = st.selectbox("Filtro Tiempo", ["Todas", "3 horas", "6 horas", "9 horas", "12 horas"])

    st.header("5. NASA FIRMS")
    api_key = st.text_input("API Key", type="password")
    test_mode = st.checkbox("Modo Prueba (Simular)", value=False)
    
    if st.button("Cargar Incendios"):
        with st.spinner("Cargando incendios..."):
            st.session_state.df_incendios = backend.obtener_incendios_nasa_real(
                map_key=api_key,
                _gdf_instalaciones=st.session_state.gdf_instalaciones,
                dias=sel_days,
                modo_prueba=test_mode
            )
            if not st.session_state.df_incendios.empty:
                st.success(f"{len(st.session_state.df_incendios)} incendios cargados.")
            else:
                st.warning("No se encontraron incendios.")

    st.header("6. Informe")
    # Placeholder for report buttons, logic will be handled later in the script but buttons rendered here
    # We need to calculate risk first to know if we can enable buttons.
    # So we might need to move the risk calculation logic BEFORE the sidebar rendering?
    # Or just render buttons here and use session state.
    
    # Actually, Streamlit runs top to bottom. If we want buttons here, we need risk state ready.
    # But risk calculation depends on filters which are also here.
    # Risk is calculated in MAIN LOGIC.
    # We can use st.empty() placeholders or just check if 'gdf_riesgo' exists in session state from PREVIOUS run?
    # No, that would be laggy.
    # Better approach: Move Risk Calculation to a function and call it before Sidebar finishes?
    # Or just render the sidebar buttons at the end of the script using st.sidebar again?
    # Yes, we can append to sidebar later.
    
    report_container = st.container()

    st.header("7. Controles")
    
    # Botón para borrar polígono
    if st.session_state.poligono_coords:
        if st.button("🗑️ Borrar Polígono"):
            st.session_state.poligono_coords = []
            st.rerun()
    
    if st.button("Resetear Todo"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

# --- MAIN LOGIC ---

# 1. Filter Data
gdf_filtered = st.session_state.gdf_instalaciones.copy()
if not gdf_filtered.empty:
    mask = (
        gdf_filtered['TIPO_EQUIPO'].astype(str).isin(sel_equipos) &
        gdf_filtered['CCAA'].astype(str).isin(sel_ccaa)
    )
    gdf_filtered = gdf_filtered[mask]
    
    if search_text:
        mask_search = (
            gdf_filtered['NAME'].astype(str).str.lower().str.contains(search_text.lower()) |
            gdf_filtered['POBLACION'].astype(str).str.lower().str.contains(search_text.lower())
        )
        gdf_filtered = gdf_filtered[mask_search]

# 2. Risk Analysis
gdf_risk = gpd.GeoDataFrame()
df_fires_filtered = pd.DataFrame()

if not st.session_state.df_incendios.empty and not gdf_filtered.empty:
    with st.spinner('⏳ Analizando riesgo de incendios...'):
        df_fires_filtered = backend.filtrar_incendios_por_tiempo(st.session_state.df_incendios, time_filter)
        
        # Calculate Proximity (CACHED)
        gdf_risk = backend.calcular_riesgo_por_proximidad(gdf_filtered, df_fires_filtered, sel_radius)
        
        # Calculate Polygon (if exists) - Convert to tuple for caching
        if st.session_state.poligono_coords:
            poligono_tuple = tuple(st.session_state.poligono_coords)
            gdf_risk = backend.calcular_riesgo_por_poligono(gdf_risk, poligono_tuple)
        else:
            gdf_risk['COMPROMETIDA_POLIGONO'] = False
            
        gdf_risk['COMPROMETIDA_FINAL'] = gdf_risk['COMPROMETIDA_DISTANCIA'] | gdf_risk['COMPROMETIDA_POLIGONO']
        st.session_state.gdf_riesgo = gdf_risk
else:
    # If no fires, risk is just polygon if drawn, or empty
    if not gdf_filtered.empty:
         gdf_risk = gdf_filtered.copy()
         gdf_risk['COMPROMETIDA_DISTANCIA'] = False
         gdf_risk['DISTANCIA_MIN_M'] = float('inf')
         gdf_risk['MAX_CONFIDENCE'] = 0
         
         if st.session_state.poligono_coords:
             poligono_tuple = tuple(st.session_state.poligono_coords)
             gdf_risk = backend.calcular_riesgo_por_poligono(gdf_risk, poligono_tuple)
             gdf_risk['COMPROMETIDA_FINAL'] = gdf_risk['COMPROMETIDA_POLIGONO']
         else:
             gdf_risk['COMPROMETIDA_POLIGONO'] = False
             gdf_risk['COMPROMETIDA_FINAL'] = False
         
         st.session_state.gdf_riesgo = gdf_risk

# 3. Map Generation
m = folium.Map(location=[40.4168, -3.7038], zoom_start=6)

# Add Fires
if not df_fires_filtered.empty:
    for _, row in df_fires_filtered.iterrows():
        color = "orange" if row['CONFIDENCE'] < 100 else "red"
        
        firms_url = row.get('FIRMS_URL', '#')
        popup_html = f"""
        <b>Incendio</b><br>
        Confianza: {row['CONFIDENCE']}%<br>
        <a href="{firms_url}" target="_blank">Ver en FIRMS</a>
        """
        
        folium.CircleMarker(
            location=[row['LAT'], row['LON']],
            radius=5,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(m)



# Add Installations
if not st.session_state.gdf_riesgo.empty:
    df_map = st.session_state.gdf_riesgo.copy()
    
    # Priorizar instalaciones afectadas primero
    if show_only_risk:
        df_map = df_map[df_map['COMPROMETIDA_FINAL']]
    
    # OPTIMIZACIÓN: Límite de 300 marcadores con prioridad inteligente
    MAX_MARKERS = 300
    
    if len(df_map) > MAX_MARKERS:
        # Separar afectadas de no afectadas
        df_afectadas = df_map[df_map.get('COMPROMETIDA_FINAL', False)]
        df_no_afectadas = df_map[~df_map.get('COMPROMETIDA_FINAL', False)]
        
        # Priorizar afectadas (todas) + las más cercanas de las no afectadas
        if len(df_afectadas) >= MAX_MARKERS:
            # Si hay más de 100 afectadas, mostrar solo las más cercanas
            df_map = df_afectadas.nsmallest(MAX_MARKERS, 'DISTANCIA_MIN_M')
            st.warning(f"⚠️ Mostrando {MAX_MARKERS} instalaciones afectadas más cercanas de {len(df_afectadas)} para mejor rendimiento")
        else:
            # Mostrar todas las afectadas + completar hasta 100 con las más cercanas
            remaining = MAX_MARKERS - len(df_afectadas)
            if not df_no_afectadas.empty and remaining > 0:
                df_cercanas = df_no_afectadas.nsmallest(remaining, 'DISTANCIA_MIN_M')
                df_map = pd.concat([df_afectadas, df_cercanas])
            else:
                df_map = df_afectadas
            
            st.info(f"ℹ️ Mostrando {len(df_afectadas)} afectadas + {len(df_map) - len(df_afectadas)} cercanas de {len(st.session_state.gdf_riesgo)} totales")
    
    # Create Clusters con configuración optimizada
    marker_cluster = MarkerCluster(
        name='Instalaciones',
        overlay=True,
        control=True,
        icon_create_function=None,
        options={
            'maxClusterRadius': 50,  # Radio máximo para agrupar (más pequeño = más clusters)
            'spiderfyOnMaxZoom': True,  # Expandir en forma de araña cuando se hace zoom
            'showCoverageOnHover': False,  # No mostrar área de cobertura
            'zoomToBoundsOnClick': True,  # Hacer zoom al cluster al hacer click
            'disableClusteringAtZoom': 15  # Desactivar clustering a partir de zoom 15
        }
    ).add_to(m)
    
    for _, row in df_map.iterrows():
        # Determine color based on risk and type
        color = "blue" # Default
        
        # Risk overrides everything
        if row.get('COMPROMETIDA_FINAL'):
            color = "red"
        else:
            # If not compromised, color by type
            tipo = str(row.get('TIPO_EQUIPO', '')).upper()
            if 'GNL' in tipo:
                color = "orange"
            elif 'GLP' in tipo:
                color = "green"
        
        # Create Icon
        icon = folium.Icon(color=color, icon="info-sign")
        
        popup_html = f"""
        <b>{row['NAME']}</b><br>
        Tipo: {row['TIPO_EQUIPO']}<br>
        CCAA: {row.get('CCAA', 'N/A')}<br>
        <a href="{row['GMAPS_URL']}" target="_blank">Ver Mapa</a><br>
        <a href="{row['STREET_URL']}" target="_blank">Street View</a>
        """
        
        folium.Marker(
            location=[row['LAT'], row['LON']],
            popup=folium.Popup(popup_html, max_width=300),
            icon=icon,
            tooltip=f"{row['NAME']} ({row['TIPO_EQUIPO']})"
        ).add_to(marker_cluster)

# Dibujar polígono persistente si existe
if st.session_state.poligono_coords:
    # Convertir coordenadas (lat, lon) a (lon, lat) para Folium
    polygon_coords_folium = [(lon, lat) for lat, lon in st.session_state.poligono_coords]
    
    folium.Polygon(
        locations=st.session_state.poligono_coords,  # Folium espera [(lat, lon), ...]
        color='blue',
        fill=True,
        fillColor='lightblue',
        fillOpacity=0.3,
        weight=2,
        popup='Área de Análisis'
    ).add_to(m)

# Draw Control
draw = Draw(
    export=False,
    position="topleft",
    draw_options={
        "polyline": False,
        "rectangle": True,
        "circle": False,
        "marker": False,
        "circlemarker": False,
        "polygon": True,
    },
)
draw.add_to(m)

# Render Map
# Usamos un key fijo para evitar remontajes innecesarios
output = st_folium(m, width="100%", height=600, key="folium_map")

# Handle Draw Output
# Lógica Estricta: Solo actualizamos si hay cambios claros
try:
    if output:
        # Caso 1: Hay nuevos dibujos
        if output.get("all_drawings"):
            last_draw = output["all_drawings"][-1]
            
            # Procesar solo si es Polígono o Rectángulo (aunque solo permitimos estos)
            if last_draw["geometry"]["type"] in ["Polygon", "Rectangle"]:
                coords = last_draw["geometry"]["coordinates"][0]
                # Convertir a (lat, lon) para backend
                # Ensure we handle potential format issues
                new_coords = []
                for c in coords:
                    if len(c) >= 2:
                        new_coords.append((c[1], c[0]))
                
                # Solo actualizar y re-ejecutar si es nuevo
                if new_coords != st.session_state.poligono_coords:
                    st.session_state.poligono_coords = new_coords
                    st.rerun()
        
        # Caso 2: El usuario borró todo explícitamente (lista vacía pero output válido)
        # output['all_drawings'] es []
        elif output.get("all_drawings") == []:
            if st.session_state.poligono_coords:
                st.session_state.poligono_coords = []
                st.rerun()

except Exception as e:
    st.error(f"Error procesando polígono: {e}")
    # Optional: Log stack trace
    # import traceback
    # st.text(traceback.format_exc())
# --- RESULTS & METRICS ---
st.subheader("Resultados")

total_inst = len(gdf_filtered)
affected_inst = len(st.session_state.gdf_riesgo[st.session_state.gdf_riesgo['COMPROMETIDA_FINAL']]) if not st.session_state.gdf_riesgo.empty else 0

m1, m2 = st.columns(2)
m1.metric("Total", total_inst)
m2.metric("Afectadas", affected_inst, delta_color="inverse")

if affected_inst > 0:
    df_display = st.session_state.gdf_riesgo[st.session_state.gdf_riesgo['COMPROMETIDA_FINAL']].copy()
    
    # Ensure URL columns exist
    if 'GMAPS_URL' not in df_display.columns: df_display['GMAPS_URL'] = None
    if 'STREET_URL' not in df_display.columns: df_display['STREET_URL'] = None
    
    df_display = df_display[['ID_INSTALACION', 'NAME', 'POBLACION', 'DISTANCIA_MIN_M', 'GMAPS_URL', 'STREET_URL']]
    df_display['Distancia (km)'] = (df_display['DISTANCIA_MIN_M'] / 1000).round(2)
    df_display = df_display.drop(columns=['DISTANCIA_MIN_M']).sort_values('Distancia (km)')
    
    st.dataframe(
        df_display, 
        hide_index=True, 
        use_container_width=True,
        column_config={
            "GMAPS_URL": st.column_config.LinkColumn("Mapa", display_text="Ver Mapa"),
            "STREET_URL": st.column_config.LinkColumn("Street View", display_text="Ver Street View")
        }
    )

# --- REPORT BUTTONS IN SIDEBAR ---
with st.sidebar:
    st.header("6. Informe")
    with st.container():
        has_results = affected_inst > 0
        
        # Prepare filters info (always needed for reports)
        filtros_info = {
            'Equipos': ", ".join(sel_equipos),
            'CCAA': ", ".join(sel_ccaa),
            'Búsqueda': search_text,
            'Radio Riesgo': sel_radius_label,
            'Días Histórico': str(sel_days),
            'Filtro Tiempo': time_filter,
            'Ver solo afectadas': "Sí" if show_only_risk else "No"
        }

        # Excel Report
        excel_buffer = io.BytesIO()
        if has_results:
            backend.guardar_excel_completo(
                excel_buffer, 
                st.session_state.gdf_riesgo, 
                st.session_state.df_incendios, 
                filtros_info
            )
        
        st.download_button(
            label="Descargar Excel",
            data=excel_buffer.getvalue() if has_results else b"",
            file_name=f"Informe_Riesgo_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            disabled=not has_results,
            use_container_width=True
        )

        # HTML Report
        if has_results:
            html_content = backend.generar_html_report(
                st.session_state.gdf_riesgo, 
                st.session_state.df_incendios, 
                filtros_info
            )
            
            st.download_button(
                label="Descargar HTML",
                data=html_content,
                file_name=f"Informe_Riesgo_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
                mime="text/html",
                disabled=not has_results,
                use_container_width=True
            )

