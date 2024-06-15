import streamlit as st

import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity



df = pd.read_csv('./processed_movies_dataset.csv')



features = df.drop(columns=['Title'])

cosine_sim = cosine_similarity(features)



def recommend_movies(title, df = df, cosine_sim = cosine_sim):

    idx = df.index[df['Title'] == title].tolist()[0]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)

    sim_scores = sim_scores[1:6]

    movie_indices = [i[0] for i in sim_scores]

return df['Title'].iloc[movie_indices]



st.title("Movie Recommendation System")



movie_name = st.selectbox('Select A Movie:', df['Title'])



if movie_name:

    recommendations = recommend_movies(movie_name)

    st.write(f"Recommendations For '{movie_name}':")

    for rec in recommendations:

        st.write(rec)