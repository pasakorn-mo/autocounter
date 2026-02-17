import streamlit as st
import pandas as pd
import yaml
from datetime import datetime, timedelta
from data.database import AutoCounterDB

# --- 1. ปรับแต่ง CSS ---
st.set_page_config(page_title="BEM AutoCounter V5", layout="wide")

st.markdown("""
<style>
    .stApp { opacity: 1 !important; }
    div[data-testid="stStatusWidget"] { display: none; }
    
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #e9ecef;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    .status-online { color: #28a745; font-weight: bold; }
    .status-offline { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

db = AutoCounterDB()

@st.cache_data
def load_config():
    with open("config/settings.yaml", "r") as f:
        return yaml.safe_load(f)

cfg = load_config()
cam_to_plaza = {str(c['id']).strip().upper(): str(c['plaza']).strip().upper() for c in cfg['cameras']}

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Filter")
    plazas_list = sorted(list(set(cam_to_plaza.values())))
    selected_plaza = st.selectbox("Select Plaza", ["ALL PLAZAS"] + plazas_list)
    st.divider()
    st.info("ส่วนสถานะและตารางวันนี้จะอัปเดตอัตโนมัติทุก 10 วินาที")

st.title("🚗 BEM AutoCounter Traffic System")

# --- 2. ส่วน Real-time (Fragment: รีเฟรชเฉพาะจุด ไม่จาง) ---
@st.fragment(run_every=10)
def render_live_updates(plaza_filter):
    # --- 2.1 Monitoring Status ---
    st.subheader(f"🖥️ สถานะระบบ: {plaza_filter}")
    all_stats = db.get_statuses()
    
    # Filter Stats
    if plaza_filter == "ALL PLAZAS":
        filtered_stats = all_stats
    else:
        filtered_stats = [s for s in all_stats if cam_to_plaza.get(str(s[0]).strip().upper()) == plaza_filter.upper()]

    cols = st.columns(4)
    with cols[0]:
        st.markdown(f'<div class="metric-card"><b>SYSTEM</b><br><span class="status-online">🟢 Running</span><br><small>{datetime.now().strftime("%H:%M:%S")}</small></div>', unsafe_allow_html=True)

    for i, (cam_id, status, last) in enumerate(filtered_stats):
        col_idx = (i + 1) % 4
        is_online = (status == "Online")
        # Check Timeout (40s)
        try:
            if (datetime.now() - datetime.fromisoformat(last)).total_seconds() > 40: is_online = False
        except: pass

        color = "status-online" if is_online else "status-offline"
        icon = "🟢" if is_online else "🔴"
        with cols[col_idx]:
            st.markdown(f'<div class="metric-card"><b>{cam_id}</b><br><span class="{color}">{icon} {status}</span><br><small>{last[11:19] if last else "--:--:--"}</small></div>', unsafe_allow_html=True)

    # --- 2.2 Today's Details by Lane ---
    st.divider()
    today_str = datetime.now().strftime('%d/%m/%Y')
    st.subheader(f"📍ปริมาณจราจร รายช่องทาง {today_str}")
    raw_data = db.get_realtime_counts()
    if raw_data:
        df = pd.DataFrame(raw_data, columns=['Plaza', 'Lane', 'Count'])
        if plaza_filter != "ALL PLAZAS":
            df = df[df['Plaza'].str.strip().str.upper() == plaza_filter.upper()]
        st.dataframe(df, width='stretch', hide_index=True)
    else:
        st.info("ไม่มีข้อมูลบันทึกวันนี้")

    # --- 2.3 Hourly Traffic Volume (กลับมาแล้ว!) ---
    st.subheader(f"⏱️ ปริมาณจราจร รายชั่วโมง {today_str}")
    h_data = db.get_hourly_counts()
    if h_data:
        dfh = pd.DataFrame(h_data, columns=['Plaza', 'Lane', 'Hour', 'Count'])
        if plaza_filter != "ALL PLAZAS":
            dfh = dfh[dfh['Plaza'].str.strip().str.upper() == plaza_filter.upper()]
        
        if not dfh.empty:
            pivot = dfh.pivot_table(index=['Plaza', 'Lane'], columns='Hour', values='Count', fill_value=0)
            # สร้างคอลัมน์ชั่วโมงให้ครบ 00-23
            all_hours = [f"{i:02d}" for i in range(24)]
            pivot = pivot.reindex(columns=all_hours, fill_value=0).astype(int)
            st.dataframe(pivot, width='stretch')
        else:
            st.info(f"No hourly records found for {plaza_filter}")
    else:
        st.info("No hourly data recorded yet.")

# รันฟังก์ชัน Fragment
render_live_updates(selected_plaza)

# --- 3. ส่วน History Search (ทำงานแยกกัน ไม่จาง) ---
st.divider()
st.subheader("📅 History Report Search")
h_c1, h_c2, h_c3 = st.columns([2, 2, 1])

with h_c1:
    d_start = st.date_input("From Date", datetime.now() - timedelta(days=1))
with h_c2:
    d_end = st.date_input("To Date", datetime.now())

if h_col3_btn := h_c3.button("🔍 Search History", type="primary"):
    with st.spinner("Querying Database..."):
        # กรองข้อมูลย้อนหลังตามช่วงวันที่ และ Plaza ที่เลือก
        res = db.get_hourly_counts_range(d_start.isoformat(), d_end.isoformat(), selected_plaza)
        if res:
            df_res = pd.DataFrame(res, columns=['Plaza', 'Lane', 'Date', 'Hour', 'Count'])
            p_res = df_res.pivot_table(index=['Date', 'Plaza', 'Lane'], columns='Hour', values='Count', fill_value=0)
            # จัดการชั่วโมงให้ครบ 00-23 สำหรับรายงาน
            all_h = [f"{i:02d}" for i in range(24)]
            p_res = p_res.reindex(columns=all_h, fill_value=0).astype(int)
            
            st.success(f"พบข้อมูลทั้งหมด {len(df_res)} รายการ")
            st.dataframe(p_res, width='stretch')
            
            # ปุ่ม Download
            csv = df_res.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV Report", csv, f"Report_{selected_plaza}.csv", "text/csv")
        else:
            st.warning("ไม่พบข้อมูลในช่วงเวลาที่เลือก")