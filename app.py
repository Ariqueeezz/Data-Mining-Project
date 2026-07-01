import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances, silhouette_score
import joblib
import io
import os

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
        "icon": "🏦",
        "desc": "Mitra korporat dengan volume transaksi dan GTV tertinggi. Penggerak utama arus kas Perusahaan X.",
        "strategi": "Key account management — dedikasikan tim khusus, negosiasi kontrak jangka panjang, dan pantau stabilitas performa secara real-time.",
    },
    1: {
        "icon": "🏪",
        "desc": "Mayoritas mitra dengan skala bisnis kecil dan margin profit tipis. Umumnya pelaku usaha mikro/kecil.",
        "strategi": "Program inkubasi dan pendampingan pertumbuhan; insentif bertingkat untuk mendorong kenaikan volume transaksi.",
    },
    2: {
        "icon": "🏢",
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
.metric-card .sub { font-size: 12px; color: #94A3B8; margin-top: 2px; }

.cluster-card {
    border-radius: 12px;
    padding: 20px 22px;
    margin-bottom: 14px;
    border-left: 5px solid;
}
.cluster-card h3 { margin: 0 0 6px 0; font-size: 16px; font-weight: 700; color: #0F172A; }
.cluster-card p  { margin: 0; font-size: 14px; color: #475569; line-height: 1.6; }
.cluster-card .strat {
    margin-top: 10px; font-size: 13px;
    background: rgba(255,255,255,0.75);
    padding: 10px 14px; border-radius: 8px; color: #1E293B;
}
.section-title {
    font-size: 18px; font-weight: 700; color: #0F172A;
    margin: 28px 0 14px 0; padding-bottom: 8px;
    border-bottom: 2px solid #E2E8F0;
}
.upload-box {
    background: #F1F5F9; border: 2px dashed #CBD5E1;
    border-radius: 16px; padding: 48px 32px; text-align: center;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt_idr(val):
    if abs(val) >= 1e12: return f"Rp {val/1e12:.2f} T"
    elif abs(val) >= 1e9: return f"Rp {val/1e9:.2f} M"
    elif abs(val) >= 1e6: return f"Rp {val/1e6:.2f} Jt"
    return f"Rp {val:,.0f}"

def fmt_trx(val):
    if val >= 1e6: return f"{val/1e6:.2f} Jt"
    elif val >= 1e3: return f"{val/1e3:.1f} Rb"
    return f"{val:,.0f}"


# ── Load model (sekali, di-cache) ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    """
    Load model K-Medoids yang sudah diexport dari Google Colab.
    Model menyimpan medoid_indices_ yang digunakan untuk assign klaster.
    """
    model_path = "model_clustering.joblib"
    if not os.path.exists(model_path):
        st.error("❌ File model_clustering.joblib tidak ditemukan. "
                 "Pastikan file model ada di direktori yang sama dengan app.py.")
        st.stop()
    return joblib.load(model_path)


# ── Predict menggunakan medoid indices dari model tersimpan ───────────────────
def predict_with_saved_model(model, df_scaled_new, df_scaled_train):
    """
    Assign klaster ke data baru menggunakan medoid_indices_ dari model tersimpan.
    
    Alur:
    1. Ambil medoid_indices_ dari model (posisi medoid di data training)
    2. Ambil vektor fitur medoid dari data training yang sudah di-scale
    3. Hitung jarak Manhattan data baru ke setiap medoid
    4. Assign ke klaster dengan jarak terdekat
    """
    medoid_indices = model.medoid_indices_          # [0, 69, 2] — index di data training
    medoid_vectors = df_scaled_train[medoid_indices] # ambil baris medoid dari data training

    # Hitung jarak Manhattan setiap baris data baru ke setiap medoid
    # Shape: (n_samples_baru, n_medoids)
    distances_to_medoids = pairwise_distances(
        df_scaled_new, medoid_vectors, metric='manhattan'
    )

    # Assign ke medoid terdekat
    labels = np.argmin(distances_to_medoids, axis=1)
    return labels


# ── Pipeline utama ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_pipeline(file_bytes):
    """
    Pipeline preprocessing + predict menggunakan model tersimpan.
    StandardScaler di-fit ke data yang diupload (sama persis seperti di Colab).
    """
    model = load_model()

    df = pd.read_excel(io.BytesIO(file_bytes))

    # Cast numerik
    for col in ['GTV FY25-A', 'GR FY25-A', 'NR FY25-A']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')

    # Feature engineering — Margin Profit (sama seperti Colab)
    df['Margin Profit (%)'] = np.where(
        df['GTV FY25-A'] != 0,
        (df['NR FY25-A'] / df['GTV FY25-A']) * 100,
        0
    )

    feature_cols = ['Trx FY25-A', 'GTV FY25-A', 'GR FY25-A', 'NR FY25-A', 'Margin Profit (%)']
    available = [c for c in feature_cols if c in df.columns]

    # StandardScaler — fit ke data upload (konsisten dengan Colab)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[available])

    # Predict menggunakan medoid dari model tersimpan
    labels = predict_with_saved_model(model, df_scaled, df_scaled)

    df['Cluster'] = labels

    # Silhouette Score
    dist_matrix = pairwise_distances(df_scaled, metric='manhattan')
    sil = silhouette_score(dist_matrix, labels, metric='precomputed')

    summary = df.groupby('Cluster')[available].mean()

    return df, summary, sil, model


def assign_labels(summary):
    """Auto-assign label berdasarkan ranking GTV rata-rata."""
    if 'GTV FY25-A' not in summary.columns:
        return {c: f"Klaster {c}" for c in summary.index}, \
               {c: "📌" for c in summary.index}, \
               {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(summary.index)}

    gtv_rank = summary['GTV FY25-A'].rank(ascending=False).astype(int)
    n = len(summary)

    rank_to_meta = {
        1: ("🏦", "Mitra Skala Raksasa", "#1A56DB"),
        n: ("🏪", "Mitra Skala Kecil",   "#E3A008"),
    }
    default_mid = ("🏢", "Mitra Skala Besar", "#057A55")

    label_map = {}; icon_map = {}; color_map = {}
    for c, rank in gtv_rank.items():
        ico, lbl, col = rank_to_meta.get(rank, default_mid)
        label_map[c] = lbl
        icon_map[c]  = ico
        color_map[c] = col

    return label_map, icon_map, color_map


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Segmentasi Mitra")
    st.markdown("**Payment Gateway — Perusahaan X**")
    st.markdown("Periode: Januari – September 2025")
    st.divider()
    st.markdown("### Upload Dataset")
    uploaded = st.file_uploader(
        "Upload file Excel (.xlsx)", type=["xlsx"],
        help="Kolom: Trx FY25-A · GTV FY25-A · GR FY25-A · NR FY25-A",
    )
    st.divider()
    st.markdown(
        "<small style='color:#94A3B8'>Proyek Data Mining<br>"
        "Raihan Ariq Muzakki · 202310715297<br>"
        "Universitas Bhayangkara Jakarta Raya · 2026</small>",
        unsafe_allow_html=True,
    )


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='font-size:26px;font-weight:800;color:#0F172A;margin-bottom:4px'>"
    "Segmentasi Kinerja Mitra Payment Gateway</h1>"
    "<p style='color:#64748B;font-size:14px;margin-top:0'>"
    "K-Medoids (PAM) · StandardScaler · Manhattan Distance · CRISP-DM</p>",
    unsafe_allow_html=True,
)

# ── Halaman awal sebelum upload ───────────────────────────────────────────────
if uploaded is None:
    st.markdown("""
    <div class="upload-box">
        <div style="font-size:48px;margin-bottom:12px">📂</div>
        <div style="font-size:18px;font-weight:700;color:#0F172A;margin-bottom:8px">
            Upload Dataset untuk Memulai Analisis
        </div>
        <div style="font-size:14px;color:#64748B;max-width:440px;margin:0 auto;line-height:1.7">
            Upload file Excel (.xlsx) melalui panel sidebar di sebelah kiri.<br>
            Pastikan file mengandung kolom:<br>
            <code>Trx FY25-A</code> · <code>GTV FY25-A</code> · 
            <code>GR FY25-A</code> · <code>NR FY25-A</code>
        </div>
        <div style="margin-top:20px;font-size:13px;color:#94A3B8">
            Model K-Medoids yang telah dilatih akan digunakan secara otomatis
            untuk mengklasifikasikan mitra ke dalam segmen yang sesuai.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Tentang Aplikasi Ini</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**🤖 Model Tersimpan**  \nMenggunakan model K-Medoids (PAM) yang telah dilatih dari Google Colab dan diekspor via `joblib`. Tidak perlu training ulang.")
    with c2:
        st.markdown("**⚙️ Preprocessing**  \nStandardScaler (Z-score) + jarak Manhattan — konsisten dengan pipeline training di Google Colab.")
    with c3:
        st.markdown("**📈 Hasil**  \nSetiap mitra dikelompokkan ke segmen terdekat berdasarkan jarak Manhattan ke medoid klaster yang tersimpan.")
    st.stop()


# ── Pipeline ──────────────────────────────────────────────────────────────────
file_bytes = uploaded.read()

with st.spinner("⚙️ Memproses data menggunakan model tersimpan..."):
    try:
        df, summary, sil_score, model = run_pipeline(file_bytes)
    except Exception as e:
        st.error(f"❌ Gagal memproses data: {e}")
        st.caption("Pastikan format kolom sudah sesuai: Trx FY25-A, GTV FY25-A, GR FY25-A, NR FY25-A")
        st.stop()

label_map, icon_map, color_map = assign_labels(summary)
cluster_ids = sorted(df['Cluster'].unique())
df['Label'] = df['Cluster'].map(label_map)

st.success(f"✅ Selesai! {len(df)} mitra berhasil disegmentasi menggunakan model K-Medoids tersimpan.")

# ── KPI ───────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card">
        <div class="label">Total Mitra</div>
        <div class="value">{len(df)}</div>
        <div class="sub">Mitra aktif yang dianalisis</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card">
        <div class="label">Jumlah Klaster</div>
        <div class="value">{model.n_clusters}</div>
        <div class="sub">Dari model tersimpan (K=3)</div>
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
dist_rows = []
for c in cluster_ids:
    n = int((df['Cluster'] == c).sum())
    dist_rows.append({
        'Klaster': f"Cluster {c}",
        'Label': label_map.get(c, f"Klaster {c}"),
        'Jumlah Mitra': n,
        'Persentase (%)': f"{n/len(df)*100:.1f}%",
    })
st.dataframe(pd.DataFrame(dist_rows), use_container_width=True, hide_index=True)


# ── Profil rata-rata ──────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Profil Rata-Rata per Klaster</div>', unsafe_allow_html=True)
summary_disp = summary.copy().reset_index()
summary_disp['Cluster'] = summary_disp['Cluster'].apply(
    lambda c: f"Cluster {c} — {label_map.get(c, '')}"
)
for col in summary_disp.columns[1:]:
    if 'Trx' in col:
        summary_disp[col] = summary_disp[col].apply(fmt_trx)
    elif 'Margin' in col:
        summary_disp[col] = summary_disp[col].apply(lambda x: f"{x:.4f}%")
    else:
        summary_disp[col] = summary_disp[col].apply(fmt_idr)
summary_disp.columns = ['Klaster'] + [c.replace(' FY25-A', '') for c in summary_disp.columns[1:]]
st.dataframe(summary_disp, use_container_width=True, hide_index=True)


# ── Chart distribusi ──────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Distribusi Visual Klaster</div>', unsafe_allow_html=True)
counts     = [int((df['Cluster'] == c).sum()) for c in cluster_ids]
bar_colors = [color_map.get(c, PALETTE[i % len(PALETTE)]) for i, c in enumerate(cluster_ids)]
xlabels    = [f"C{c}\n{label_map.get(c,'')[:10]}" for c in cluster_ids]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.patch.set_facecolor('#F8FAFC')

bars = axes[0].bar(xlabels, counts, color=bar_colors, width=0.5, edgecolor='white', linewidth=1.5)
for bar, count in zip(bars, counts):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                 str(count), ha='center', va='bottom', fontweight='700', fontsize=12)
axes[0].set_title('Jumlah Mitra per Klaster', fontweight='700', fontsize=13, pad=10)
axes[0].set_ylabel('Jumlah Mitra')
axes[0].set_facecolor('#F8FAFC')
axes[0].spines[['top', 'right']].set_visible(False)

pie_labels = [f"C{c}: {label_map.get(c,'')[:8]}\n{n/len(df)*100:.1f}%"
              for c, n in zip(cluster_ids, counts)]
axes[1].pie(counts, labels=pie_labels, colors=bar_colors, startangle=90,
            wedgeprops=dict(width=0.55, edgecolor='white', linewidth=2),
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
        ax.scatter(df.loc[mask, 'GTV FY25-A'], df.loc[mask, 'NR FY25-A'],
                   color=color_map.get(c, PALETTE[c % len(PALETTE)]),
                   alpha=0.75, s=90, edgecolors='white', linewidths=0.8, zorder=3,
                   label=f"C{c} — {label_map.get(c, f'Klaster {c}')}")
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


# ── Box Plot ──────────────────────────────────────────────────────────────────
if 'Margin Profit (%)' in df.columns:
    st.markdown('<div class="section-title">Box Plot — Margin Profit per Klaster</div>', unsafe_allow_html=True)
    fig3, ax = plt.subplots(figsize=(8, 4))
    fig3.patch.set_facecolor('#F8FAFC')
    ax.set_facecolor('#F8FAFC')
    bp = ax.boxplot(
        [df[df['Cluster'] == c]['Margin Profit (%)'].values for c in cluster_ids],
        patch_artist=True, notch=False,
        medianprops=dict(color='white', linewidth=2.5),
        whiskerprops=dict(linewidth=1.5), capprops=dict(linewidth=1.5),
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
    lbl = label_map.get(c, f"Klaster {c}")
    ico = icon_map.get(c, meta["icon"])
    col = color_map.get(c, "#64748B")
    n_anggota = int((df['Cluster'] == c).sum())
    pct = n_anggota / len(df) * 100
    st.markdown(f"""
    <div class="cluster-card" style="background:{col}12;border-left-color:{col}">
        <h3>{ico} Cluster {c} — {lbl}
            <span style="font-size:13px;font-weight:500;color:#64748B;margin-left:8px">
                n={n_anggota} ({pct:.1f}%)
            </span>
        </h3>
        <p>{meta['desc']}</p>
        <div class="strat">💡 <strong>Rekomendasi:</strong> {meta['strategi']}</div>
    </div>
    """, unsafe_allow_html=True)


# ── Tabel lengkap + download ──────────────────────────────────────────────────
st.markdown('<div class="section-title">Data Mitra Lengkap dengan Label Klaster</div>', unsafe_allow_html=True)
display_cols = [c for c in df.columns if c not in ['Cluster', 'Label']] + ['Cluster', 'Label']
st.dataframe(df[[c for c in display_cols if c in df.columns]], use_container_width=True, hide_index=True)

csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="⬇️ Download Hasil Segmentasi (.csv)",
    data=csv,
    file_name="hasil_segmentasi_mitra_payment_gateway.csv",
    mime="text/csv",
)
