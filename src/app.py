import os
import streamlit as st 
from agent import Agent
import folium
from folium.plugins import MarkerCluster, MeasureControl
from streamlit_folium import folium_static
import logging
import json
from dotenv import load_dotenv

import base64
from pathlib import Path

load_dotenv()

google_api_key = os.environ.get("GOOGLE_API_KEY")

if google_api_key:
    print(f"GOOGLE_API_KEY loaded com sucesso")
else:
    print("GOOGLE_API_KEY não está definida nas variáveis de ambiente.")

LOGGER = logging.getLogger(__name__)

st.set_page_config(page_title="Meu App Incrível", layout="wide")

st.markdown("""
<style>
    /* Altera o fundo do campo de texto (text_area) */
    div[data-testid="stTextArea"] textarea {
        background-color: #E8F0FE; /* Um tom de azul claro */
        color: #000000; /* Cor do texto digitado */
    }

    /* Altera a aparência do botão */
    div[data-testid="stButton"] > button {
        background-color: #0047AB; /* Azul cobalto */
        color: white; /* Cor do texto do botão */
        border: none;
        padding: 12px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px; /* Bordas arredondadas */
    }
</style>
""", unsafe_allow_html=True)

### Initialization
def initialize_session_state():
    if "marker" not in st.session_state:
        st.session_state.marker = []

def initialize_map(center, zoom=10):
    if "map" not in st.session_state or st.session_state.map is None:
        st.session_state.center = center
        st.session_state.zoom = zoom
        folium_map = folium.Map(
            location=center,
            zoom_start=zoom)
        st.session_state.map = folium_map
    return st.session_state.map

initialize_session_state()
### Dashboard
# st.image("icon_house.png", width=20)


col1, col2 = st.columns([0.05, 1]) # Ajuste os valores para controlar a largura das colunas

with col1:
    st.image("icon_house.png", width=60)

with col2:
    st.markdown("# Welcome to the Real State Agent App!") # Use o nível de cabeçalho desejado

st.write("This app will help you find a good place live with detailed tips and information about the neighborhood.")


def get_image_as_base64(file):
    """Lê um arquivo de imagem e o converte para uma string Base64."""
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_path = "logo.png"

# st.title("\n Welcome to the Real State Agent App!")
st.markdown("<br>", unsafe_allow_html=True) # Duas quebras de linha para um espaço maior


# if not Path(img_path).is_file():
#     st.error("Arquivo de imagem não encontrado! Verifique o caminho: " + img_path)
# else:
#     img_base64 = get_image_as_base64(img_path)
#     st.markdown(f"""
#         <img src="data:image/png;base64,{img_base64}" width="1200" height="100">
#         """,
#         unsafe_allow_html=True
#     )

points_coordinates = []

agent = Agent(google_api_key)

col1, col2 = st.columns(2)

with col1:
    request = st.text_area("Where would you like to live? Please describe your request in Portuguese.")
    button = st.button("Get neighborhood information")
    
    box = st.container(height=300, border=True)
    with box:
        container = st.empty()
        container.header("Neighborhood Information")

if button and request:
    points = agent.get_tips(request)
    try:
        container.write(points["spots"])
    except KeyError:
        container.write("""
        Sorry, I couldn't generate an anything for your request.
        Please try again with a different request.
        """)
    try:
        coordinates = points["coordinates"]
        if not coordinates:
            raise ValueError("Coordenadas estão vazias.")

        if coordinates.startswith("```json"):
            coordinates = coordinates.strip("`")  # remove todas as crases
            lines = coordinates.split("\n")
            coordinates = "\n".join(lines[1:-1])  # remove a primeira e última linha
        if isinstance(coordinates, str):
            coordinates = json.loads(coordinates)

        points_coordinates = []
        
        print(points_coordinates)
        for loc in coordinates.get("locations", []):
            points_coordinates.append(
                [loc["lat"], loc["lon"]]
                )
        st.session_state['marker'] = [
            folium.Marker(
                location=point,
                icon=folium.Icon(color="red", icon="info-sign", prefix="glyphicon")
            )
            for point in points_coordinates
        ]
    
    except KeyError:
        LOGGER.warning("No coordinates found in the itinerary response.")
    except json.JSONDecodeError as e:
        LOGGER.error(f"Erro ao decodificar JSON: {e}")
    except Exception as e:
        LOGGER.error(f"Erro inesperado: {e}")
    st.session_state['points_coordinates'] = points_coordinates  
    
with col2:
    folium_map = initialize_map(center=[ -23.5612, -46.6559], zoom=12)
    fg = folium.FeatureGroup(name="Markers")

    for marker in st.session_state['marker']:
        fg.add_child(marker)
    fg.add_to(folium_map)

    if st.session_state.get("points_coordinates"):
        folium_map.fit_bounds(st.session_state["points_coordinates"])

    folium_static(folium_map, width=450, height=500)