import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances, silhouette_score
import kmedoids
import io

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Segmentasi Mitra Payment Gateway",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
COLORS = {
    "raksasa": "#1A56DB",   # Cluster 0 — Mitra Skala Raksasa
    "kecil":   "#E3A008",   # Cluster 1 — Mitra Skala Kecil
    "besar":   "#057A55",   # Cluster 2 — Mitra Skala Besar
}
PALETTE = [COLORS["raksasa"], COLORS["kecil"], COLORS["besar"]]

CLUSTER_META = {
    0: {
        "label": "Mitra Skala Raksasa",
        "icon": "🏦",
        "color": COLORS["raksasa"],
        "desc": "Mitra korporat dengan volume transaksi dan GTV tertinggi. "
                "Penggerak utama arus kas Perusahaan X.",
        "strategi": "Key account management — dedikasikan tim khusus, "
                    "negosiasi kontrak jangka panjang, dan pantau stabilitas "
                    "performa secara real-time.",
    },
    1: {
        "label": "Mitra Skala Kecil",
        "icon": "🏪",
        "color": COLORS["kecil"],
        "desc": "Mayoritas mitra (97,3%) dengan skala bisnis kecil dan margin "
                "profit tipis. Umumnya pelaku usaha mikro/kecil.",
        "strategi": "Program inkubasi dan pendampingan pertumbuhan; insentif "
                    "bertingkat untuk mendorong kenaikan volume transaksi.",
    },
    2: {
        "label": "Mitra Skala Besar",
        "icon": "🏢",
        "color": COLORS["besar"],
        "desc": "Mitra dengan margin profit tertinggi (0,98%) — profil mitra "
                "ideal dengan efisiensi pendapatan bersih terbaik.",
        "strategi": "Perlakuan istimewa setara key account; program loyalitas "
                    "eksklusif dan eksplorasi perluasan layanan strategis.",
    },
}

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=DM+Mono&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.metric-card {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
}
.metric-card .label {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748B;
    margin-bottom: 4px;
}
.metric-card .value {
    font-size: 28px;
    font-weight: 700;
    color: #0F172A;
    line-height: 1.2;
}
.metric-card .sub {
    font-size: 13px;
    color: #94A3B8;
    margin-top: 2px;
}

.cluster-card {
    border-radius: 12px;
    padding: 20px 22px;
    margin-bottom: 16px;
    border-left: 5px solid;
}
.cluster-card h3 { margin: 0 0 6px 0; font-size: 16px; font-weight: 700; }
.cluster-card p  { margin: 0; font-size: 14px; color: #475569; line-height: 1.6; }
.cluster-card .strat {
    margin-top: 10px;
    font-size: 13px;
    background: rgba(255,255,255,0.7);
    padding: 10px 14px;
    border-radius: 8px;
    color: #1E293B;
}

.section-title {
    font-size: 20px;
    font-weight: 700;
    color: #0F172A;
    margin: 32px 0 16px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid #E2E8F0;
}
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt_idr(val):
    """Format angka ke format Rupiah singkat."""
    if abs(val) >= 1e12:
        return f"Rp {val/1e12:.2f} T"
    elif abs(val) >= 1e9:
        return f"Rp {val/1e9:.2f} M"
    elif abs(val) >= 1e6:
        return f"Rp {val/1e6:.2f} Jt"
    else:
        return f"Rp {val:,.0f}"

def fmt_trx(val):
    if val >= 1e6:
        return f"{val/1e6:.2f} Jt"
    elif val >= 1e3:
        return f"{val/1e3:.1f} Rb"
    return f"{val:,.0f}"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Segmentasi Mitra")
    st.markdown("**Payment Gateway — Perusahaan X**")
    st.markdown("Periode: Januari – September 2025")
    st.divider()
    st.markdown("### Upload Dataset")
    uploaded = st.file_uploader(
        "Upload file Excel (.xlsx)",
        type=["xlsx"],
        help="Pastikan file mengandung kolom: Trx FY25-A, GTV FY25-A, GR FY25-A, NR FY25-A",
    )
    st.divider()
    st.markdown("### Parameter Model")
    n_clusters = st.slider("Jumlah Klaster (K)", min_value=2, max_value=6, value=3)
    st.caption("Default K=3 berdasarkan hasil evaluasi Elbow + Silhouette Score (0,9183)")
    st.divider()
    st.markdown(
        "<small style='color:#94A3B8'>Proyek Data Mining · Raihan Ariq Muzakki<br>"
        "Universitas Bhayangkara Jakarta Raya · 2026</small>",
        unsafe_allow_html=True,
    )


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='font-size:28px;font-weight:800;color:#0F172A;margin-bottom:4px'>"
    "Segmentasi Kinerja Mitra Payment Gateway</h1>"
    "<p style='color:#64748B;font-size:15px;margin-top:0'>Algoritma K-Medoids (PAM) · "
    "StandardScaler + Manhattan Distance · CRISP-DM</p>",
    unsafe_allow_html=True,
)

# ── Load / process data ───────────────────────────────────────────────────────
@st.cache_data
def process_data(file_bytes, k):
    df = pd.read_excel(io.BytesIO(file_bytes))

    # Cast numerik
    for col in ['GTV FY25-A', 'GR FY25-A', 'NR FY25-A']:
        df[col] = df[col].astype('int64')

    # Feature engineering
    df['Margin Profit (%)'] = np.where(
        df['GTV FY25-A'] != 0,
        (df['NR FY25-A'] / df['GTV FY25-A']) * 100,
        0
    )

    # Scaling
    num_cols = ['Trx FY25-A', 'GTV FY25-A', 'GR FY25-A', 'NR FY25-A', 'Margin Profit (%)']
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[num_cols])

    # Distance matrix + K-Medoids
    dist_matrix = pairwise_distances(df_scaled, metric='manhattan')
    model = kmedoids.KMedoids(n_clusters=k, random_state=42, method='pam')
    df['Cluster'] = model.fit_predict(dist_matrix)

    # Silhouette
    score = silhouette_score(dist_matrix, df['Cluster'], metric='precomputed')

    # Label
    # Tentukan label berdasarkan GTV rata-rata per klaster
    cluster_gtv = df.groupby('Cluster')['GTV FY25-A'].mean().sort_values(ascending=False)
    label_map = {}
    labels_pool = ["Mitra Skala Raksasa", "Mitra Skala Besar", "Mitra Skala Kecil"]
    for i, (cl, _) in enumerate(cluster_gtv.items()):
        label_map[cl] = labels_pool[i] if i < len(labels_pool) else f"Klaster {cl}"
    df['Label'] = df['Cluster'].map(label_map)

    summary = df.groupby('Cluster')[num_cols].mean()

    return df, summary, score, dist_matrix


if uploaded is None:
    # ── Demo state (data summary hardcoded dari hasil aktual) ─────────────────
    st.info("📂 Upload file Excel di sidebar untuk menjalankan analisis secara langsung. "
            "Berikut tampilan hasil berdasarkan dataset aktual Perusahaan X.")
    st.divider()

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class="metric-card">
            <div class="label">Total Mitra Dianalisis</div>
            <div class="value">146</div>
            <div class="sub">Mitra aktif Jan–Sep 2025</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card">
            <div class="label">Jumlah Klaster</div>
            <div class="value">K = 3</div>
            <div class="sub">Elbow Method + Silhouette</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card">
            <div class="label">Silhouette Score</div>
            <div class="value">0,9183</div>
            <div class="sub">Kualitas sangat baik (maks 1,0)</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="metric-card">
            <div class="label">Metode Normalisasi</div>
            <div class="value">Z-Score</div>
            <div class="sub">StandardScaler · Manhattan</div>
        </div>""", unsafe_allow_html=True)

    # ── Profil klaster ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Profil Segmen Mitra</div>', unsafe_allow_html=True)

    demo_summary = pd.DataFrame({
        'Klaster': ['Cluster 0', 'Cluster 1', 'Cluster 2'],
        'Label': ['Mitra Skala Raksasa', 'Mitra Skala Kecil', 'Mitra Skala Besar'],
        'Jumlah Mitra': [3, 142, 1],
        'Persen (%)': ['2,1%', '97,3%', '0,7%'],
        'Rata-rata TRX': ['21,98 Jt', '126 Rb', '9,77 Jt'],
        'Rata-rata GTV': ['Rp 2,21 T', 'Rp 924,42 Jt', 'Rp 52,88 M'],
        'Rata-rata NR': ['Rp 11,17 M', 'Rp 10,98 Jt', 'Rp 5,20 M'],
        'Margin Profit': ['0,52%', '0,18%', '0,98%'],
    })
    st.dataframe(demo_summary, use_container_width=True, hide_index=True)

    # ── Cluster cards ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Interpretasi & Rekomendasi Strategi</div>', unsafe_allow_html=True)
    for cid, meta in CLUSTER_META.items():
        bg = meta["color"] + "15"
        border = meta["color"]
        st.markdown(f"""
        <div class="cluster-card" style="background:{bg};border-left-color:{border}">
            <h3>{meta['icon']} Cluster {cid} — {meta['label']}</h3>
            <p>{meta['desc']}</p>
            <div class="strat">💡 <strong>Rekomendasi:</strong> {meta['strategi']}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Chart: distribusi anggota ──────────────────────────────────────────────
    st.markdown('<div class="section-title">Distribusi Anggota per Klaster</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#F8FAFC')

    # Bar chart
    labels_demo = ['C0\nRaksasa', 'C1\nKecil', 'C2\nBesar']
    counts_demo = [3, 142, 1]
    bars = axes[0].bar(labels_demo, counts_demo, color=PALETTE, width=0.5, edgecolor='white', linewidth=1.5)
    for bar, count in zip(bars, counts_demo):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                     str(count), ha='center', va='bottom', fontweight='700', fontsize=13)
    axes[0].set_title('Jumlah Mitra per Klaster', fontweight='700', fontsize=13, pad=12)
    axes[0].set_ylabel('Jumlah Mitra')
    axes[0].set_facecolor('#F8FAFC')
    axes[0].spines[['top', 'right']].set_visible(False)

    # Pie chart
    pcts = [3/146*100, 142/146*100, 1/146*100]
    wedge_props = dict(width=0.55, edgecolor='white', linewidth=2)
    axes[1].pie(
        pcts, labels=['Raksasa\n2,1%', 'Kecil\n97,3%', 'Besar\n0,7%'],
        colors=PALETTE, autopct='', startangle=90,
        wedgeprops=wedge_props, textprops={'fontsize': 11, 'fontweight': '600'}
    )
    axes[1].set_title('Komposisi Mitra (%)', fontweight='700', fontsize=13, pad=12)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Scatter plot demo ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Visualisasi Klaster — GTV vs Margin Profit</div>', unsafe_allow_html=True)
    st.caption("Upload dataset untuk melihat scatter plot aktual. Berikut ilustrasi distribusi klaster berdasarkan karakteristik profil tiap segmen.")

    np.random.seed(42)
    demo_points = {
        0: {'gtv': np.random.normal(2.21e12, 2e11, 3),  'margin': np.random.normal(0.52, 0.05, 3)},
        1: {'gtv': np.random.normal(9.24e8,  3e8,  142), 'margin': np.random.normal(0.18, 0.08, 142)},
        2: {'gtv': np.array([5.29e10]),                  'margin': np.array([0.98])},
    }
    fig2, ax = plt.subplots(figsize=(10, 5))
    fig2.patch.set_facecolor('#F8FAFC')
    ax.set_facecolor('#F8FAFC')
    for cid, pts in demo_points.items():
        ax.scatter(pts['gtv'], pts['margin'],
                   color=PALETTE[cid], alpha=0.75, s=80 if cid==1 else 160,
                   edgecolors='white', linewidths=1, zorder=3,
                   label=f"C{cid} — {CLUSTER_META[cid]['label']}")
    ax.set_xlabel('GTV (Gross Transaction Value)', fontsize=12)
    ax.set_ylabel('Margin Profit (%)', fontsize=12)
    ax.set_title('Distribusi Klaster: GTV vs Margin Profit', fontsize=14, fontweight='700', pad=12)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: fmt_idr(x)))
    ax.legend(loc='upper left', framealpha=0.9, fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

else:
    # ── Live analysis dengan file upload ──────────────────────────────────────
    file_bytes = uploaded.read()
    with st.spinner("Menjalankan K-Medoids clustering..."):
        try:
            df, summary, sil_score, dist_matrix = process_data(file_bytes, n_clusters)
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses data: {e}")
            st.stop()

    st.success(f"✅ Clustering selesai! {len(df)} mitra berhasil disegmentasi ke dalam {n_clusters} klaster.")

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Total Mitra</div>
            <div class="value">{len(df)}</div>
            <div class="sub">Mitra aktif yang dianalisis</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Jumlah Klaster</div>
            <div class="value">K = {n_clusters}</div>
            <div class="sub">Parameter model aktif</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Silhouette Score</div>
            <div class="value">{sil_score:.4f}</div>
            <div class="sub">{'Sangat Baik ✓' if sil_score > 0.7 else 'Baik ✓' if sil_score > 0.5 else 'Cukup'}</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Metode</div>
            <div class="value">K-Medoids</div>
            <div class="sub">PAM · Manhattan · Z-Score</div>
        </div>""", unsafe_allow_html=True)

    # Tabel summary
    st.markdown('<div class="section-title">Profil Rata-Rata per Klaster</div>', unsafe_allow_html=True)
    summary_display = summary.copy()
    summary_display.index.name = 'Cluster'
    summary_display['Trx FY25-A'] = summary_display['Trx FY25-A'].apply(lambda x: fmt_trx(x))
    summary_display['GTV FY25-A'] = summary_display['GTV FY25-A'].apply(fmt_idr)
    summary_display['GR FY25-A']  = summary_display['GR FY25-A'].apply(fmt_idr)
    summary_display['NR FY25-A']  = summary_display['NR FY25-A'].apply(fmt_idr)
    summary_display['Margin Profit (%)'] = summary_display['Margin Profit (%)'].apply(lambda x: f"{x:.2f}%")
    summary_display.columns = ['Rata-rata TRX', 'Rata-rata GTV', 'Rata-rata GR', 'Rata-rata NR', 'Margin Profit']
    st.dataframe(summary_display, use_container_width=True)

    # Distribusi klaster
    st.markdown('<div class="section-title">Distribusi Anggota per Klaster</div>', unsafe_allow_html=True)
    dist_df = df['Cluster'].value_counts().sort_index().reset_index()
    dist_df.columns = ['Cluster', 'Jumlah Mitra']
    dist_df['Persentase (%)'] = (dist_df['Jumlah Mitra'] / len(df) * 100).round(1).astype(str) + '%'
    st.dataframe(dist_df, use_container_width=True, hide_index=True)

    # Scatter plot
    st.markdown('<div class="section-title">Visualisasi Klaster — GTV vs NR</div>', unsafe_allow_html=True)
    fig3, ax = plt.subplots(figsize=(10, 5))
    fig3.patch.set_facecolor('#F8FAFC')
    ax.set_facecolor('#F8FAFC')
    colors_map = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(sorted(df['Cluster'].unique()))}
    for cluster_id in sorted(df['Cluster'].unique()):
        mask = df['Cluster'] == cluster_id
        ax.scatter(df.loc[mask, 'GTV FY25-A'], df.loc[mask, 'NR FY25-A'],
                   color=colors_map[cluster_id], alpha=0.7, s=80,
                   edgecolors='white', linewidths=0.8,
                   label=f'Cluster {cluster_id}', zorder=3)
    ax.set_xlabel('GTV (Gross Transaction Value)', fontsize=12)
    ax.set_ylabel('NR (Net Revenue)', fontsize=12)
    ax.set_title('Distribusi Klaster: GTV vs Net Revenue', fontsize=14, fontweight='700', pad=12)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: fmt_idr(x)))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: fmt_idr(x)))
    ax.legend(fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=0.25, linestyle='--')
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    # Box plot
    st.markdown('<div class="section-title">Distribusi Margin Profit per Klaster</div>', unsafe_allow_html=True)
    fig4, ax = plt.subplots(figsize=(8, 4))
    fig4.patch.set_facecolor('#F8FAFC')
    ax.set_facecolor('#F8FAFC')
    cluster_order = sorted(df['Cluster'].unique())
    bp = ax.boxplot(
        [df[df['Cluster'] == c]['Margin Profit (%)'].values for c in cluster_order],
        patch_artist=True, notch=False,
        medianprops=dict(color='white', linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker='o', markersize=4, alpha=0.5),
    )
    for patch, color in zip(bp['boxes'], [colors_map[c] for c in cluster_order]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_xticklabels([f'Cluster {c}' for c in cluster_order])
    ax.set_ylabel('Margin Profit (%)', fontsize=12)
    ax.set_title('Distribusi Margin Profit per Klaster', fontsize=13, fontweight='700', pad=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()

    # Data tabel lengkap
    st.markdown('<div class="section-title">Data Mitra dengan Label Klaster</div>', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Download Hasil Clustering (.csv)",
        data=csv,
        file_name="hasil_segmentasi_mitra.csv",
        mime="text/csv",
    )
