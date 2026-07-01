import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances, silhouette_score
import kmedoids
import io

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Segmentasi Mitra Payment Gateway",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
PALETTE = ["#1A56DB", "#E3A008", "#057A55", "#9333EA", "#DC2626", "#0891B2"]

CLUSTER_META_DEFAULT = {
    0: {
        "label": "Mitra Skala Raksasa",
        "icon": "🏦",
        "color": "#1A56DB",
        "desc": "Mitra korporat dengan volume transaksi dan GTV tertinggi. Penggerak utama arus kas Perusahaan X.",
        "strategi": "Key account management — dedikasikan tim khusus, negosiasi kontrak jangka panjang, dan pantau stabilitas performa secara real-time.",
    },
    1: {
        "label": "Mitra Skala Kecil",
        "icon": "🏪",
        "color": "#E3A008",
        "desc": "Mayoritas mitra dengan skala bisnis kecil dan margin profit tipis. Umumnya pelaku usaha mikro/kecil.",
        "strategi": "Program inkubasi dan pendampingan pertumbuhan; insentif bertingkat untuk mendorong kenaikan volume transaksi.",
    },
    2: {
        "label": "Mitra Skala Besar",
        "icon": "🏢",
        "color": "#057A55",
        "desc": "Mitra dengan margin profit tertinggi — profil mitra ideal dengan efisiensi pendapatan bersih terbaik.",
        "strategi": "Perlakuan istimewa setara key account; program loyalitas eksklusif dan eksplorasi perluasan layanan strategis.",
    },
}

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.metric-card {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
}
.metric-card .label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #64748B;
    margin-bottom: 4px;
}
.metric-card .value {
    font-size: 26px;
    font-weight: 700;
    color: #0F172A;
    line-height: 1.2;
}
.metric-card .sub {
    font-size: 12px;
    color: #94A3B8;
    margin-top: 2px;
}
.cluster-card {
    border-radius: 12px;
    padding: 20px 22px;
    margin-bottom: 14px;
    border-left: 5px solid;
}
.cluster-card h3 { margin: 0 0 6px 0; font-size: 16px; font-weight: 700; color: #0F172A; }
.cluster-card p  { margin: 0; font-size: 14px; color: #475569; line-height: 1.6; }
.cluster-card .strat {
    margin-top: 10px;
    font-size: 13px;
    background: rgba(255,255,255,0.75);
    padding: 10px 14px;
    border-radius: 8px;
    color: #1E293B;
}
.section-title {
    font-size: 18px;
    font-weight: 700;
    color: #0F172A;
    margin: 28px 0 14px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid #E2E8F0;
}
.upload-box {
    background: #F1F5F9;
    border: 2px dashed #CBD5E1;
    border-radius: 16px;
    padding: 48px 32px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt_idr(val):
    if abs(val) >= 1e12:
        return f"Rp {val/1e12:.2f} T"
    elif abs(val) >= 1e9:
        return f"Rp {val/1e9:.2f} M"
    elif abs(val) >= 1e6:
        return f"Rp {val/1e6:.2f} Jt"
    return f"Rp {val:,.0f}"

def fmt_trx(val):
    if val >= 1e6:
        return f"{val/1e6:.2f} Jt"
    elif val >= 1e3:
        return f"{val/1e3:.1f} Rb"
    return f"{val:,.0f}"

def get_color(cluster_id, n_clusters):
    return PALETTE[cluster_id % len(PALETTE)]


# ── Core pipeline ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_clustering(file_bytes, k):
    """
    Pipeline lengkap:
    1. Load Excel
    2. Feature engineering (Margin Profit)
    3. StandardScaler (Z-score)
    4. Pairwise distance Manhattan
    5. K-Medoids PAM
    6. Silhouette Score
    """
    df = pd.read_excel(io.BytesIO(file_bytes))

    # Cast numerik
    for col in ['GTV FY25-A', 'GR FY25-A', 'NR FY25-A']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')

    # Feature engineering — Margin Profit
    df['Margin Profit (%)'] = np.where(
        df['GTV FY25-A'] != 0,
        (df['NR FY25-A'] / df['GTV FY25-A']) * 100,
        0
    )

    # Kolom fitur clustering
    feature_cols = ['Trx FY25-A', 'GTV FY25-A', 'GR FY25-A', 'NR FY25-A', 'Margin Profit (%)']
    available = [c for c in feature_cols if c in df.columns]

    # StandardScaler
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[available])

    # Distance matrix (Manhattan)
    dist_matrix = pairwise_distances(df_scaled, metric='manhattan')

    # K-Medoids PAM — retrain setiap upload
    model = kmedoids.KMedoids(n_clusters=k, random_state=42, method='pam')
    labels = model.fit_predict(dist_matrix)
    df['Cluster'] = labels

    # Silhouette Score
    sil = silhouette_score(dist_matrix, labels, metric='precomputed')

    # Summary per klaster
    summary = df.groupby('Cluster')[available].mean()

    return df, summary, sil, model, available


def assign_labels(df, summary):
    """
    Otomatis assign label berdasarkan rata-rata GTV per klaster:
    Tertinggi = Raksasa, Terendah = Kecil, Sisanya = Besar/Menengah
    """
    if 'GTV FY25-A' not in summary.columns:
        label_map = {c: f"Klaster {c}" for c in summary.index}
        return df, label_map

    gtv_rank = summary['GTV FY25-A'].rank(ascending=False).astype(int)
    n = len(summary)
    label_pool = {
        1: ("🏦", "Mitra Skala Raksasa",  "#1A56DB"),
        n: ("🏪", "Mitra Skala Kecil",    "#E3A008"),
    }
    default_mid = ("🏢", "Mitra Skala Besar", "#057A55")

    label_map   = {}
    icon_map    = {}
    color_map   = {}

    for cluster_id, rank in gtv_rank.items():
        icon, lbl, col = label_pool.get(rank, default_mid)
        label_map[cluster_id] = lbl
        icon_map[cluster_id]  = icon
        color_map[cluster_id] = col

    df['Label'] = df['Cluster'].map(label_map)
    return df, label_map, icon_map, color_map


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Segmentasi Mitra")
    st.markdown("**Payment Gateway — Perusahaan X**")
    st.markdown("Periode: Januari – September 2025")
    st.divider()

    st.markdown("### ① Upload Dataset")
    uploaded = st.file_uploader(
        "Upload file Excel (.xlsx)",
        type=["xlsx"],
        help="Kolom yang dibutuhkan: Trx FY25-A, GTV FY25-A, GR FY25-A, NR FY25-A",
    )

    st.divider()
    st.markdown("### ② Parameter Model")
    n_clusters = st.slider(
        "Jumlah Klaster (K)", min_value=2, max_value=6, value=3,
        help="K=3 adalah hasil optimal berdasarkan Elbow Method + Silhouette Score (0,9183)"
    )
    st.caption(f"Model akan dijalankan ulang dengan K={n_clusters} setiap kali file diupload.")

    st.divider()
    st.markdown(
        "<small style='color:#94A3B8'>Proyek Data Mining<br>"
        "Raihan Ariq Muzakki · 202310715297<br>"
        "Universitas Bhayangkara Jakarta Raya · 2026</small>",
        unsafe_allow_html=True,
    )


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='font-size:26px;font-weight:800;color:#0F172A;margin-bottom:4px'>"
    "Segmentasi Kinerja Mitra Payment Gateway</h1>"
    "<p style='color:#64748B;font-size:14px;margin-top:0'>"
    "K-Medoids (PAM) · StandardScaler · Manhattan Distance · CRISP-DM</p>",
    unsafe_allow_html=True,
)

# ── State: belum upload ───────────────────────────────────────────────────────
if uploaded is None:
    st.markdown("""
    <div class="upload-box">
        <div style="font-size:48px;margin-bottom:12px">📂</div>
        <div style="font-size:18px;font-weight:700;color:#0F172A;margin-bottom:8px">
            Upload Dataset untuk Memulai Analisis
        </div>
        <div style="font-size:14px;color:#64748B;max-width:420px;margin:0 auto">
            Upload file Excel (.xlsx) melalui panel sidebar di sebelah kiri.<br>
            Pastikan file mengandung kolom:<br>
            <code>Trx FY25-A</code> · <code>GTV FY25-A</code> · 
            <code>GR FY25-A</code> · <code>NR FY25-A</code>
        </div>
        <div style="margin-top:20px;font-size:13px;color:#94A3B8">
            Model K-Medoids akan berjalan otomatis setelah file berhasil diupload.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Tentang Aplikasi Ini</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""**🔬 Algoritma**  
K-Medoids PAM dengan matriks jarak Manhattan — tahan terhadap outlier dan data keuangan yang berdistribusi *skewed*.""")
    with col2:
        st.markdown("""**⚙️ Preprocessing**  
StandardScaler (Z-score normalization) untuk menyeimbangkan variabel dengan rentang nilai yang sangat berbeda.""")
    with col3:
        st.markdown("""**📈 Evaluasi**  
Silhouette Score sebagai metrik kualitas klaster. Nilai K=3 menghasilkan skor 0,9183 (sangat baik).""")
    st.stop()


# ── State: file terupload → jalankan pipeline ─────────────────────────────────
file_bytes = uploaded.read()

with st.spinner("⚙️ Menjalankan K-Medoids clustering..."):
    try:
        df, summary, sil_score, model, feature_cols = run_clustering(file_bytes, n_clusters)
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat memproses data: {e}")
        st.caption("Pastikan format kolom sudah sesuai: Trx FY25-A, GTV FY25-A, GR FY25-A, NR FY25-A")
        st.stop()

# Assign label otomatis
df, label_map, icon_map, color_map = assign_labels(df, summary)
cluster_ids = sorted(df['Cluster'].unique())
colors_list = [color_map.get(c, PALETTE[i % len(PALETTE)]) for i, c in enumerate(cluster_ids)]

st.success(f"✅ Selesai! {len(df)} mitra berhasil disegmentasi ke dalam {n_clusters} klaster.")

# ── KPI row ───────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card">
        <div class="label">Total Mitra</div>
        <div class="value">{len(df)}</div>
        <div class="sub">Mitra aktif yang dianalisis</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card">
        <div class="label">Jumlah Klaster (K)</div>
        <div class="value">{n_clusters}</div>
        <div class="sub">Parameter model aktif</div>
    </div>""", unsafe_allow_html=True)
with c3:
    quality = "Sangat Baik ✓" if sil_score > 0.7 else "Baik ✓" if sil_score > 0.5 else "Cukup"
    st.markdown(f"""<div class="metric-card">
        <div class="label">Silhouette Score</div>
        <div class="value">{sil_score:.4f}</div>
        <div class="sub">{quality}</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card">
        <div class="label">Metode</div>
        <div class="value">K-Medoids</div>
        <div class="sub">PAM · Manhattan · Z-Score</div>
    </div>""", unsafe_allow_html=True)


# ── Distribusi anggota ────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Distribusi Anggota per Klaster</div>', unsafe_allow_html=True)

dist_data = []
for c in cluster_ids:
    n = (df['Cluster'] == c).sum()
    dist_data.append({
        'Klaster': f"Cluster {c}",
        'Label': label_map.get(c, f"Klaster {c}"),
        'Jumlah Mitra': n,
        'Persentase (%)': f"{n/len(df)*100:.1f}%",
    })
dist_df = pd.DataFrame(dist_data)
st.dataframe(dist_df, use_container_width=True, hide_index=True)


# ── Profil rata-rata per klaster ──────────────────────────────────────────────
st.markdown('<div class="section-title">Profil Rata-Rata per Klaster</div>', unsafe_allow_html=True)

summary_display = summary.copy().reset_index()
summary_display['Cluster'] = summary_display['Cluster'].apply(
    lambda c: f"Cluster {c} — {label_map.get(c, '')}"
)
for col in summary_display.columns[1:]:
    if 'Trx' in col:
        summary_display[col] = summary_display[col].apply(fmt_trx)
    elif 'Margin' in col:
        summary_display[col] = summary_display[col].apply(lambda x: f"{x:.4f}%")
    else:
        summary_display[col] = summary_display[col].apply(fmt_idr)

summary_display.columns = ['Klaster'] + [c.replace(' FY25-A', '') for c in summary_display.columns[1:]]
st.dataframe(summary_display, use_container_width=True, hide_index=True)


# ── Visualisasi: Bar + Pie ────────────────────────────────────────────────────
st.markdown('<div class="section-title">Distribusi Visual Klaster</div>', unsafe_allow_html=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.patch.set_facecolor('#F8FAFC')

counts  = [int((df['Cluster'] == c).sum()) for c in cluster_ids]
xlabels = [f"C{c}\n{label_map.get(c,'')[:10]}" for c in cluster_ids]
bar_colors = [color_map.get(c, PALETTE[i]) for i, c in enumerate(cluster_ids)]

# Bar
bars = axes[0].bar(xlabels, counts, color=bar_colors, width=0.5, edgecolor='white', linewidth=1.5)
for bar, count in zip(bars, counts):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(count), ha='center', va='bottom', fontweight='700', fontsize=12)
axes[0].set_title('Jumlah Mitra per Klaster', fontweight='700', fontsize=13, pad=10)
axes[0].set_ylabel('Jumlah Mitra')
axes[0].set_facecolor('#F8FAFC')
axes[0].spines[['top', 'right']].set_visible(False)

# Donut
pie_labels = [f"C{c}: {label_map.get(c,'')[:8]}\n{n/len(df)*100:.1f}%" 
              for c, n in zip(cluster_ids, counts)]
axes[1].pie(counts, labels=pie_labels, colors=bar_colors,
            startangle=90, wedgeprops=dict(width=0.55, edgecolor='white', linewidth=2),
            textprops={'fontsize': 10, 'fontweight': '600'})
axes[1].set_title('Komposisi Mitra (%)', fontweight='700', fontsize=13, pad=10)

plt.tight_layout()
st.pyplot(fig)
plt.close()


# ── Scatter Plot ──────────────────────────────────────────────────────────────
if 'GTV FY25-A' in df.columns and 'NR FY25-A' in df.columns:
    st.markdown('<div class="section-title">Scatter Plot — GTV vs Net Revenue</div>', unsafe_allow_html=True)

    fig2, ax = plt.subplots(figsize=(10, 5))
    fig2.patch.set_facecolor('#F8FAFC')
    ax.set_facecolor('#F8FAFC')

    for c in cluster_ids:
        mask = df['Cluster'] == c
        ax.scatter(
            df.loc[mask, 'GTV FY25-A'],
            df.loc[mask, 'NR FY25-A'],
            color=color_map.get(c, PALETTE[c % len(PALETTE)]),
            alpha=0.75, s=90, edgecolors='white', linewidths=0.8, zorder=3,
            label=f"C{c} — {label_map.get(c, f'Klaster {c}')}"
        )

    ax.set_xlabel('GTV (Gross Transaction Value)', fontsize=12)
    ax.set_ylabel('NR (Net Revenue)', fontsize=12)
    ax.set_title('Distribusi Klaster: GTV vs Net Revenue', fontsize=13, fontweight='700', pad=12)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: fmt_idr(x)))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: fmt_idr(x)))
    ax.legend(fontsize=10, framealpha=0.9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=0.25, linestyle='--')
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()


# ── Box Plot Margin Profit ────────────────────────────────────────────────────
if 'Margin Profit (%)' in df.columns:
    st.markdown('<div class="section-title">Box Plot — Margin Profit per Klaster</div>', unsafe_allow_html=True)

    fig3, ax = plt.subplots(figsize=(8, 4))
    fig3.patch.set_facecolor('#F8FAFC')
    ax.set_facecolor('#F8FAFC')

    bp = ax.boxplot(
        [df[df['Cluster'] == c]['Margin Profit (%)'].values for c in cluster_ids],
        patch_artist=True, notch=False,
        medianprops=dict(color='white', linewidth=2.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker='o', markersize=4, alpha=0.4),
    )
    for patch, c in zip(bp['boxes'], cluster_ids):
        patch.set_facecolor(color_map.get(c, PALETTE[c % len(PALETTE)]))
        patch.set_alpha(0.85)

    ax.set_xticklabels([f"C{c}\n{label_map.get(c,'')}" for c in cluster_ids], fontsize=10)
    ax.set_ylabel('Margin Profit (%)', fontsize=12)
    ax.set_title('Distribusi Margin Profit per Klaster', fontsize=13, fontweight='700', pad=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()


# ── Interpretasi & Rekomendasi ────────────────────────────────────────────────
st.markdown('<div class="section-title">Interpretasi & Rekomendasi Strategi per Segmen</div>', unsafe_allow_html=True)

for c in cluster_ids:
    meta = CLUSTER_META_DEFAULT.get(c, {
        "icon": "📌",
        "desc": f"Segmen klaster {c} berdasarkan profil TRX, GTV, GR, NR, dan Margin Profit.",
        "strategi": "Lakukan analisis lebih mendalam terhadap karakteristik mitra dalam segmen ini.",
    })
    lbl   = label_map.get(c, f"Klaster {c}")
    ico   = icon_map.get(c, meta["icon"])
    col   = color_map.get(c, "#64748B")
    bg    = col + "12"
    n_anggota = int((df['Cluster'] == c).sum())
    pct   = n_anggota / len(df) * 100

    st.markdown(f"""
    <div class="cluster-card" style="background:{bg};border-left-color:{col}">
        <h3>{ico} Cluster {c} — {lbl}
            <span style="font-size:13px;font-weight:500;color:#64748B;margin-left:8px">
                n={n_anggota} ({pct:.1f}%)
            </span>
        </h3>
        <p>{meta['desc']}</p>
        <div class="strat">💡 <strong>Rekomendasi Strategi:</strong> {meta['strategi']}</div>
    </div>
    """, unsafe_allow_html=True)


# ── Tabel data lengkap ────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Data Mitra Lengkap dengan Label Klaster</div>', unsafe_allow_html=True)

display_cols = [c for c in df.columns if c != 'Cluster'] + ['Cluster', 'Label']
display_cols = [c for c in display_cols if c in df.columns]

st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

# Download
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="⬇️ Download Hasil Segmentasi (.csv)",
    data=csv,
    file_name="hasil_segmentasi_mitra_payment_gateway.csv",
    mime="text/csv",
)
