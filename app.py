import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(
    page_title="Clustering Minat Mahasiswa",
    layout="wide"
)

st.title("Clustering Minat Mahasiswa Menggunakan K-Means")

uploaded_file = st.file_uploader(
    "Upload Dataset CSV",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Awal")
    st.dataframe(df)

    # Hilangkan kolom identitas
    data = df.drop(columns=["Nama", "NIM"])

    # Ubah menjadi biner
    data_binary = data.notna().astype(int)

    st.subheader("Data Setelah Transformasi Biner")
    st.dataframe(data_binary)

    st.sidebar.header("Pengaturan K-Means")

    k = st.sidebar.slider(
        "Jumlah Cluster (K)",
        min_value=2,
        max_value=10,
        value=3
    )

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    clusters = model.fit_predict(data_binary)

    hasil = df.copy()
    hasil["Cluster"] = clusters

    st.subheader("Hasil Clustering")
    st.dataframe(hasil)

    score = silhouette_score(
        data_binary,
        clusters
    )

    st.success(
        f"Silhouette Score : {score:.4f}"
    )

    st.subheader("Distribusi Cluster")

    fig, ax = plt.subplots(figsize=(8,5))

    sns.countplot(
        x="Cluster",
        data=hasil,
        ax=ax
    )

    ax.set_title("Jumlah Mahasiswa per Cluster")

    st.pyplot(fig)

    st.subheader("Karakteristik Setiap Cluster")

    karakteristik = (
        data_binary
        .assign(Cluster=clusters)
        .groupby("Cluster")
        .mean()
    )

    st.dataframe(karakteristik)

    fig2, ax2 = plt.subplots(figsize=(10,6))

    sns.heatmap(
        karakteristik,
        annot=True,
        cmap="Blues"
    )

    ax2.set_title("Heatmap Karakteristik Cluster")

    st.pyplot(fig2)

    st.subheader("Kesimpulan")

    for c in sorted(hasil["Cluster"].unique()):

        jumlah = len(
            hasil[hasil["Cluster"] == c]
        )

        st.write(
            f"Cluster {c}: {jumlah} mahasiswa"
        )