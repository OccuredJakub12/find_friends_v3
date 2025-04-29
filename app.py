import json
import streamlit as st
import pandas as pd
from pycaret.clustering import load_model, predict_model  # type: ignore
import plotly.express as px  # type: ignore

MODEL_NAME = 'welcome_survey_clustering_pipeline_v1'

DATA = 'welcome_survey_simple_v2.csv'

CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v1.json'

@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
        return json.loads(f.read())

@st.cache_data
def get_all_participants():
    model = get_model()
    all_df = pd.read_csv(DATA, sep=';')
    df_with_clusters = predict_model(model, data=all_df)

    return df_with_clusters

with st.sidebar:
    st.header('Powiedz nam coś o sobie')
    st.markdown("Pomożemy Ci znaleźć osoby, które mają podobne zainteresowania")
    age = st.selectbox("Wiek", ['<18', '25-34', '45-54', '35-44', '18-24', '>=65', '55-64', 'unknown'])
    edu_level = st.selectbox("Wykształcenie", ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox("Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    fav_place = st.selectbox("Ulubione miejsce", ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.radio("Płeć", ['Mężczyzna', 'Kobieta'])
    
    person_df = pd.DataFrame([
        {
            'age': age,
            'edu_level': edu_level,
            'fav_animals': fav_animals,
            'fav_place': fav_place,
            'gender': gender
        }
    ])

model = get_model()
all_df = get_all_participants()
cluster_names_and_descriptions = get_cluster_names_and_descriptions()

predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]



tab1, tab2 = st.tabs(["🎯 Mój wynik", "📖 Opisy wszystkich grup"])

with tab1:
    # tu cały Twój obecny kod: predykcja, podobni ludzie, wykresy

    st.header(f"Najbliżej Ci do grupy {predicted_cluster_data['name']}:")
    st.markdown(predicted_cluster_data['description'])
    same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]
    st.metric("**Liczba twoich znajomych**:", len(same_cluster_df))




    # Dodajmy funkcję do porównania podobieństwa
    def calculate_similarity(row, user_row):
        matches = 0
        total = len(user_row.columns)
        for col in user_row.columns:
            if row[col] == user_row[col].values[0]:
                matches += 1
        return matches / total  # zwraca wartość od 0 do 1


    # Filtrujemy osoby z tego samego klastra
    same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id].copy()

    # Dodajemy kolumnę z podobieństwem
    same_cluster_df["similarity"] = same_cluster_df.apply(lambda row: calculate_similarity(row, person_df), axis=1)

    # Sortujemy i wybieramy top 10
    top_10_similar = same_cluster_df.sort_values(by="similarity", ascending=False).head(10)
    top_10_similar["similarity"] = (top_10_similar["similarity"] * 100).round(1).astype(str) + "%"

    # Wyświetlanie wyników w Streamlit

    st.markdown("**Top 10 osób najbardziej podobnych do Ciebie:**")
    st.table(top_10_similar[["age", "edu_level", "fav_animals", "fav_place", "gender", "similarity"]])



    st.header("Osoby z grupy")

    with st.expander("Pokaż wykres wieku"):
        age_counts = same_cluster_df["age"].value_counts().reset_index()
        age_counts.columns = ["age", "count"]
        fig = px.pie(age_counts, names="age", values="count", title="Rozkład wieku w grupie")
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig)

    with st.expander("Pokaż wykres wykształcenia"):
        edu_counts = same_cluster_df["edu_level"].value_counts().reset_index()
        edu_counts.columns = ["edu_level", "count"]
        fig = px.pie(edu_counts, names="edu_level", values="count", title="Rozkład wykształcenia w grupie")
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig)

    with st.expander("Pokaż wykres ulubionych zwierząt"):
        animals_counts = same_cluster_df["fav_animals"].value_counts().reset_index()
        animals_counts.columns = ["fav_animals", "count"]
        fig = px.pie(animals_counts, names="fav_animals", values="count", title="Rozkład ulubionych zwierząt w grupie")
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig)

    with st.expander("Pokaż wykres ulubionych miejsc"):
        place_counts = same_cluster_df["fav_place"].value_counts().reset_index()
        place_counts.columns = ["fav_place", "count"]
        fig = px.pie(place_counts, names="fav_place", values="count", title="Rozkład ulubionych miejsc w grupie")
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig)

    with st.expander("Pokaż wykres płci"):
        gender_counts = same_cluster_df["gender"].value_counts().reset_index()
        gender_counts.columns = ["gender", "count"]
        fig = px.pie(gender_counts, names="gender", values="count", title="Rozkład płci w grupie")
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig)
with tab2:
    st.header("📖 Opisy wszystkich grup")

    # Przejdź przez każdy klaster w 'cluster_names_and_descriptions'
    for cluster_id, cluster_data in cluster_names_and_descriptions.items():
        # Przypisz odpowiednią emotkę na podstawie danych grupy
        group_members = all_df[all_df["Cluster"] == cluster_id]

        # Wiek: Przypisz emotki w zależności od grup wiekowych
        age_groups = group_members['age'].value_counts().to_dict()

        # Słownik przypisujący emoji do grup wiekowych
        age_groups_emoji = {
            '<18': "🧒", 
            '18-24': "🧑", 
            '25-34': "👨", 
            '35-44': "👩", 
            '45-54': "🧔", 
            '55-64': "👴", 
            '>=65': "👵", 
            'unknown': "❓"
        }

        # Wybierz największą grupę wiekową
        most_common_age_group = max(age_groups, key=age_groups.get)
        age_emoji = age_groups_emoji.get(most_common_age_group, "❓")

        # Wykształcenie: Przypisz emotkę w zależności od poziomu wykształcenia
        if group_members["edu_level"].value_counts().idxmax() == "Podstawowe":
            edu_emoji = "📚"  # Emotka książki dla osób z podstawowym wykształceniem
        elif group_members["edu_level"].value_counts().idxmax() == "Średnie":
            edu_emoji = "🎓"  # Emotka akademicka dla średniego wykształcenia
        else:
            edu_emoji = "💼"  # Emotka pracy dla wykształcenia wyższego

        # Ulubione zwierzę: Przypisz emotkę na podstawie najczęściej wybieranego zwierzęcia
        fav_animal_counts = group_members["fav_animals"].value_counts()
        fav_animal = fav_animal_counts.idxmax() if not fav_animal_counts.empty else "Brak"

        if fav_animal == "Koty":
            animal_emoji = "🐱"
        elif fav_animal == "Psy":
            animal_emoji = "🐶"
        elif fav_animal == "Koty i Psy":
            animal_emoji = "🐱🐶"
        else:
            animal_emoji = "🐾"  # Inne zwierzęta

        # Ulubione miejsce: Przypisz emotkę w zależności od najczęściej wybieranego miejsca
        fav_place = group_members["fav_place"].value_counts().idxmax()
        if "Nad wodą" in fav_place:
            place_emoji = "🌊"
        elif "W górach" in fav_place:
            place_emoji = "⛰️"
        elif "W lesie" in fav_place:
            place_emoji = "🌳"
        else:
            place_emoji = "🏙️"  # Miejskie miejsce

        gender_count = group_members["gender"].value_counts(normalize=True)

        if gender_count.get("Mężczyzna", 0) > 0.7:
            gender_emoji = "♂️"  # Dominacja mężczyzn
        elif gender_count.get("Kobieta", 0) > 0.7:
            gender_emoji = "♀️"  # Dominacja kobiet
        else:
            gender_emoji = "⚧️"  # Różnorodność lub brak dominacji

        # Ostateczna emotka: Łączenie emotek
        emoji = f"{age_emoji} {edu_emoji} {animal_emoji} {place_emoji} {gender_emoji}"

        with st.expander(f"{emoji} {cluster_data['name']}"):
            st.write(cluster_data['description'])

            # Filtrowanie wszystkich danych uczestników (nie zależne od bieżącego użytkownika)
            all_members_in_group = all_df[all_df["Cluster"] == cluster_id]

            # Wyświetlanie liczby członków grupy
            st.write(f"Liczba członków grupy {cluster_data['name']}: {len(all_members_in_group)}")

            if len(all_members_in_group) > 0:
                st.dataframe(all_members_in_group[['age', 'edu_level', 'fav_animals', 'fav_place', 'gender']])
            else:
                st.write("Brak członków w tej grupie.")
