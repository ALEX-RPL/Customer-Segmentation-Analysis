import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler

# ===========================================
# KONFIGURASI HALAMAN
# ===========================================
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide"
)

# ===========================================
# FUNGSI UTAMA
# ===========================================
def main():
    try:
        # ===========================================
        # LOAD DATA & MODEL
        # ===========================================
        df = pd.read_csv("Mall_Customers.csv")
        kmeans = joblib.load('kmeans_model.pkl')
        
        # ===========================================
        # PREPROCESSING & PREDIKSI CLUSTER
        # ===========================================
        # Encoding gender
        df = pd.get_dummies(df, columns=['Gender'], drop_first=True)
        
        # Scaling fitur
        scaler = StandardScaler()
        features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
        scaled_data = scaler.fit_transform(df[features])
        
        # Prediksi cluster
        df['Cluster'] = kmeans.predict(scaled_data)

        # ===========================================
        # SIDEBAR
        # ===========================================
        with st.sidebar:
            st.header("⚙️ Pengaturan")
            selected_cluster = st.selectbox(
                "Pilih Cluster:",
                options=sorted(df['Cluster'].unique()),
                help="Pilih kelompok pelanggan yang ingin dianalisis"
            )
            show_raw_data = st.checkbox("Tampilkan Data Mentah")

        # ===========================================
        # VISUALISASI UTAMA
        # ===========================================
        st.title("🛍️ Customer Segmentation Analysis")
        st.markdown("---")
        
        # Plot Cluster
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("Segmentasi Berdasarkan Pendapatan & Skor Belanja")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(
                data=df,
                x='Annual Income (k$)',
                y='Spending Score (1-100)',
                hue='Cluster',
                palette='viridis',
                s=100,
                alpha=0.8,
                ax=ax
            )
            ax.set_title("Distribusi Cluster Pelanggan", fontsize=14)
            ax.set_xlabel("Pendapatan Tahunan (k$)", fontsize=12)
            ax.set_ylabel("Skor Belanja", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)

        with col2:
            st.subheader("📊 Metrik Cluster")
            cluster_data = df[df['Cluster'] == selected_cluster]
            
            st.metric(
                label="Jumlah Pelanggan",
                value=len(cluster_data)
            )
            st.metric(
                label="Rata-rata Usia",
                value=f"{cluster_data['Age'].mean():.1f} Tahun"
            )
            st.metric(
                label="Rata-rata Pendapatan",
                value=f"${cluster_data['Annual Income (k$)'].mean():.2f}k"
            )
            st.metric(
                label="Rata-rata Skor Belanja",
                value=f"{cluster_data['Spending Score (1-100)'].mean():.1f}"
            )

        # ===========================================
        # ANALISIS DETAIL
        # ===========================================
        st.markdown("---")
        st.subheader("📈 Analisis Detail")

        tab1, tab2 = st.tabs(["Distribusi Usia", "Data Pelanggan"])
        
        with tab1:
            fig_age = plt.figure(figsize=(8, 4))
            sns.histplot(
                cluster_data['Age'], 
                bins=15, 
                kde=True,
                color='teal'
            )
            plt.title(f"Distribusi Usia - Cluster {selected_cluster}")
            plt.xlabel("Usia")
            st.pyplot(fig_age)
        
        with tab2:
            st.dataframe(
                cluster_data.drop('Cluster', axis=1),
                height=300,
                use_container_width=True
            )

        # ===========================================
        # TAMPILKAN DATA MENTAH
        # ===========================================
        if show_raw_data:
            st.markdown("---")
            st.subheader("📄 Data Mentah")
            st.dataframe(df, use_container_width=True)

    except FileNotFoundError:
        st.error("""
        ❌ File tidak ditemukan! Pastikan:
        1. File `Mall_Customers.csv` ada di folder yang sama
        2. File `kmeans_model.pkl` sudah di-generate
        """)
    except Exception as e:
        st.error(f"Terjadi error: {str(e)}")

# ===========================================
# JALANKAN APLIKASI
# ===========================================
if __name__ == "__main__":
    main()