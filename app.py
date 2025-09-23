import json
import streamlit as st
import pandas as pd  # type: ignore
from pycaret.clustering import load_model, predict_model  # type: ignore
import plotly.express as px  # type: ignore
from dotenv import dotenv_values
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance

MODEL_NAME = 'welcome_survey_clustering_pipeline_v3'
DATA = 'welcome_survey_simple_v2.csv'
CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v3.json'

QDRANT_COLLECTION_NAME = "user_feedback"
VECTOR_SIZE = 1


### Secrets using Streamlit Cloud Mechanism
# https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management
env = dotenv_values(".env")
if 'QDRANT_URL' in st.secrets:
    env['QDRANT_URL'] = st.secrets['QDRANT_URL']
if 'QDRANT_API_KEY' in st.secrets:
    env['QDRANT_API_KEY'] = st.secrets['QDRANT_API_KEY']


@st.cache_resource
def get_qdrant_client():
    client = QdrantClient(
        url=env["QDRANT_URL"],
        api_key=env["QDRANT_API_KEY"],
    )
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config={"size": VECTOR_SIZE, "distance": Distance.COSINE},
        )
    return client

def add_note_to_db():
    feedback_comment = st.session_state.get("feedback_textarea", "").strip()
    if not feedback_comment:
        return
    client = get_qdrant_client()
    count = client.count(collection_name=QDRANT_COLLECTION_NAME, exact=True).count
    point = PointStruct(
        id=count + 1,
        vector=[0.0],
        payload={"text": feedback_comment},
    )
    client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=[point])

@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
        return json.loads(f.read())

@st.cache_data
def get_all_participants():
    all_df = pd.read_csv(DATA, sep=';')
    return predict_model(model, data=all_df)

def reset_feedback_on_cluster_change(new_cluster_id):
    if 'current_cluster_id' not in st.session_state:
        st.session_state.current_cluster_id = new_cluster_id
    elif st.session_state.current_cluster_id != new_cluster_id:
        st.session_state.current_cluster_id = new_cluster_id
        st.session_state.feedback_choice = None
        st.session_state.feedback_comment_submitted = False
        if 'feedback_textarea' in st.session_state:
            del st.session_state.feedback_textarea

st.set_page_config(page_title="Apka o uczestnikach", layout="centered")

with st.sidebar:
    st.header("Powiedz nam coś o sobie")
    st.markdown("Pomożemy Ci znaleźć osoby, które mają podobne zainteresowania")
    age = st.selectbox("Wiek", ['<18','18-24','25-34','35-44','45-54','55-64','>=65','unknown'])
    edu_level = st.selectbox("Wykształcenie", ['Podstawowe','Średnie','Wyższe'])
    fav_animals = st.selectbox("Ulubione zwierzęta", ['Brak ulubionych','Psy','Koty','Inne','Koty i Psy'])
    fav_place = st.selectbox("Ulubione miejsce", ['Nad wodą','W lesie','W górach','Inne'])
    gender = st.radio("Płeć", ['Mężczyzna','Kobieta'])
    person_df = pd.DataFrame([{
        'age': age,
        'edu_level': edu_level,
        'fav_animals': fav_animals,
        'fav_place': fav_place,
        'gender': gender,
    }])

model = get_model()
all_df = get_all_participants()
cluster_names_and_descriptions = get_cluster_names_and_descriptions()

predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]

reset_feedback_on_cluster_change(predicted_cluster_id)

st.header(f"Najbliżej Ci do grupy :  {predicted_cluster_data['name']}")
st.markdown(
    f"<p style='font-size:16px; line-height:1.4;'>{predicted_cluster_data['description']}</p>",
    unsafe_allow_html=True
)
st.markdown("#### Czy ten opis pasuje do Twojej tożsamości?")

if 'feedback_choice' not in st.session_state:
    st.session_state.feedback_choice = None

col1, col2 = st.columns(2)
if st.session_state.feedback_choice is None:
    with col1:
        if st.button("✅ Opis pasuje do mnie"):
            st.session_state.feedback_choice = "pasuje"
            st.rerun()
    with col2:
        if st.button("❌ Opis nie pasuje do mnie"):
            st.session_state.feedback_choice = "nie_pasuje"
            st.rerun()
else:
    if st.session_state.feedback_choice == "pasuje":
        with col1:
            st.success("✅ Wybrałeś: Opis pasuje do mnie")
        st.info("Dziękujemy za feedback. Cieszę się, że trafiliśmy w punkt! 😊")
    else:
        with col2:
            st.success("❌ Wybrałeś: Opis nie pasuje do mnie")
        if not st.session_state.get('feedback_comment_submitted', False):
            st.info("Dziękujemy za feedback. Pomóż ulepszyć nasz algorytm poprzez dodanie komentarza!")
            feedback_comment = st.text_area(
                "Podziel się swoimi uwagami",
                placeholder="Co moglibyśmy poprawić w opisie grupy?",
                key="feedback_textarea"
            )
            if st.button("📤 Wyślij uwagi", key="submit_feedback"):
                if feedback_comment.strip():
                    add_note_to_db()
                    st.session_state.feedback_comment_submitted = True
                    st.rerun()
                else:
                    st.warning("Proszę wpisać uwagi przed wysłaniem.")
        else:
            st.info("✅ Dzięki za dodatkowe uwagi! Twój feedback został zapisany.")

if st.session_state.feedback_choice is not None:
    if st.button("🔄 Zmień wybór", type="secondary"):
        st.session_state.feedback_choice = None
        st.session_state.feedback_comment_submitted = False
        if 'feedback_textarea' in st.session_state:
            del st.session_state.feedback_textarea
        st.rerun()

same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]
st.metric("Liczba twoich znajomych", len(same_cluster_df))

st.subheader("Dane osób z Twojej grupy")
fig = px.histogram(same_cluster_df.sort_values("age"), x="age")
fig.update_layout(
    title="Rozkład wieku",
    xaxis_title="Wiek",
    yaxis_title="Liczba ankietowanych",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="edu_level")
fig.update_layout(
    title="Rozkład wykształcenia",
    xaxis_title="Wykształcenie",
    yaxis_title="Liczba ankietowanych",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_animals")
fig.update_layout(
    title="Rozkład ulubionych zwierząt",
    xaxis_title="Ulubione zwierzęta",
    yaxis_title="Liczba ankietowanych",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="fav_place")
fig.update_layout(
    title="Rozkład ulubionych miejsc",
    xaxis_title="Ulubione miejsce",
    yaxis_title="Liczba ankietowanych",
)
st.plotly_chart(fig)

fig = px.histogram(same_cluster_df, x="gender")
fig.update_layout(
    title="Rozkład płci",
    xaxis_title="Płeć",
    yaxis_title="Liczba ankietowanych",
)
st.plotly_chart(fig)

st.header("Dane dotyczące wszystkich ankietowanych")

cluster_mapping = {cluster_id: data['name'] for cluster_id, data in cluster_names_and_descriptions.items()}
all_df_with_names = all_df.copy()
all_df_with_names['Cluster_Name'] = all_df_with_names['Cluster'].map(cluster_mapping)
fig = px.histogram(all_df_with_names, x="Cluster_Name")
fig.update_layout(
    title="Rozkład ankietowanych",
    xaxis_title="Wybrana grupa",
    yaxis_title="Liczebność",
)
st.plotly_chart(fig)

