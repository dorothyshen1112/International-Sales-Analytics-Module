import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# --- 1. 頁面配置 ---
st.set_page_config(page_title="雙美全球銷售分析系統", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF; color: #2D3436; }
    .stMetric { background-color: #F8F9FA; border-radius: 10px; padding: 15px; border: 1px solid #E9ECEF; }
    [data-testid="stMetricValue"] { font-size: 1.6rem !important; color: #0984E3; font-weight: 700; }
    h1, h2, h3 { color: #0984E3; font-family: 'Helvetica Neue', sans-serif; }
    div.stTabs [data-baseweb="tab-list"] { gap: 15px; border-bottom: 2px solid #F1F3F5; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 智慧資料正規化引擎 (強化防錯版) ---
def smart_normalize(df):
    if df.empty: return pd.DataFrame()
    
    # 預抓 BX 欄位 (第76欄)
    bx_note = df.iloc[:, 75].astype(str).replace('nan', '') if df.shape[1] >= 76 else None
    
    # 清理欄位名稱：轉大寫、去空格
    df.columns = [str(c).replace(' ', '').replace('\n', '').upper() for c in df.columns]
    
    mapping = {
        'DATE': ['銷貨日期', '單據日期', '日期', 'DATE', '銷貨日期A', '單據日'],
        'COUNTRY': ['國家', '地區', 'COUNTRY', '區域', '廠別名稱'],
        'CUSTOMER': ['客戶簡稱', '客戶', '客戶名稱', 'CUSTOMER', '送貨客戶全名'],
        'PRODUCT': ['品名', '業務品名', '產品名稱', 'PRODUCT', '品號', '業務規格'],
        'QTY': ['銷貨數量', '數量', '計價數量', '總數量', 'QTY', '件數'],
        'PRICE': ['單價', '單價NT', 'PRICE', '單   價'],
        'AMOUNT': ['本幣未稅金額', '本幣合計', '金額', '未稅金額', 'AMOUNT', '總計', '本幣未稅金額_1'],
        'NOTE': ['備註', '備註1', 'REMARK', '說明', '備    註']
    }

    def find_best(targets):
        for t in targets:
            if t.upper() in df.columns: return t.upper()
        return None

    p_df = pd.DataFrame()
    
    # 安全提取函數
    def safe_get(key, is_num=False):
        col = find_best(mapping[key])
        if col:
            if is_num:
                return pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
            return df[col]
        return 0 if is_num else None

    # 開始填充標準化表格
    c_date_col = find_best(mapping['DATE'])
    p_df['銷貨日期'] = pd.to_datetime(df[c_date_col], errors='coerce') if c_date_col else pd.Timestamp.now()
    
    raw_country = safe_get('COUNTRY')
    p_df['國家'] = raw_country.fillna('台灣').astype(str) if raw_country is not None else '台灣'
    
    # 區域劃分邏輯 (台灣/中國/海外)
    def classify_region(c):
        c_str = str(c).strip()
        if c_str in ['台灣', '臺灣', 'TAIWAN']: return '台灣市場'
        if c_str in ['大陸', '中國', 'CHINA', '中 國']: return '中國市場'
        return '海外市場'
    p_df['市場區域'] = p_df['國家'].apply(classify_region)
    
    raw_cust = safe_get('CUSTOMER')
    p_df['客戶'] = raw_cust.fillna('未知客戶') if raw_cust is not None else '未知客戶'
    
    raw_prod = safe_get('PRODUCT')
    p_df['產品'] = raw_prod.astype(str).str.replace('Sunmax DeusaDerm', 'Sunmax Deusaderm VITAL', case=False) if raw_prod is not None else "未知產品"
    
    p_df['數量'] = safe_get('QTY', True)
    p_df['單價'] = safe_get('PRICE', True)
    p_df['金額'] = safe_get('AMOUNT', True)
    
    # 備註處理
    if bx_note is not None and bx_note.str.len().sum() > 0:
        p_df['備註'] = bx_note
    else:
        raw_note = safe_get('NOTE')
        p_df['備註'] = raw_note.astype(str).replace('nan', '') if raw_note is not None else ""

    p_df['年度'] = p_df['銷貨日期'].dt.year
    p_df['月份'] = p_df['銷貨日期'].dt.month
    
    # FOC 分類
    kw = {
        '培訓與實操用針': 'Workshop|實操|示範|培訓|講義|Demo',
        '醫師酬勞針': '酬勞|講師|醫師|陳咸伸|王柏鈞|技術費',
        '市場贊助與樣品': 'Sponsorship|Window|Sample|樣品|贊助|Influencer|廣告',
        '客訴補償': 'complaints|客訴|補償|瑕疵|更換|due to customer'
    }
    for label, pattern in kw.items():
        p_df[label] = 0
        is_foc = (p_df['單價'] == 0) | (p_df['金額'] == 0)
        p_df.loc[is_foc & p_df['備註'].str.contains(pattern, na=False, case=False), label] = p_df['數量']
    
    p_df['FOC總量'] = p_df[list(kw.keys())].sum(axis=1)
    p_df['收費量'] = np.where(p_df['金額'] > 0, p_df['數量'], 0)
    
    return p_df

# --- 3. 側邊欄 ---
with st.sidebar:
    st.title("⚙️ 數據管理")
    admin_mode = st.toggle("管理員模式 (導入數據)")
    uploaded_file = None
    if admin_mode:
        pwd = st.text_input("管理密碼", type="password")
        if pwd == "sunmax888":
            uploaded_file = st.file_uploader("上傳 Excel", type=["xlsx"])

st.title("📊 雙美全球銷售數據指揮中心")

if uploaded_file:
    # 讀取所有 Sheet
    all_sheets = pd.read_excel(uploaded_file, sheet_name=None)
    all_data_list = []
    for name, df_temp in all_sheets.items():
        if not df_temp.empty:
            norm_df = smart_normalize(df_temp)
            if not norm_df.empty:
                all_data_list.append(norm_df)
    
    if not all_data_list:
        st.error("❌ 檔案內容為空或無法辨識欄位。")
        st.stop()
        
    df_all = pd.concat(all_data_list, ignore_index=True)
    
    # --- 關鍵修正：解決篩選連動 BUG ---
    st.info("💡 數據已整合。請設定下方條件進行分析：")
    sc1, sc2, sc3 = st.columns([1, 1, 2])
    
    # A. 區域選擇 (台灣/中國/海外)
    area_options = ['台灣市場', '中國市場', '海外市場']
    sel_area = sc1.multiselect("🏙️ 區域", options=area_options, default=['海外市場', '中國市場'])
    
    # B. 根據區域動態更新國家清單
    filtered_by_area = df_all[df_all['市場區域'].isin(sel_area)]
    available_countries = sorted(filtered_by_area['國家'].unique())
    sel_countries = sc3.multiselect("📍 國家", options=available_countries, default=available_countries)
    
    # C. 年度選擇
    yrs = sorted([int(y) for y in df_all['年度'].dropna().unique()])
    sel_yrs = sc2.multiselect("📅 年度", yrs, default=yrs[-2:] if len(yrs)>1 else yrs)

    # 最終篩選數據
    df = df_all[(df_all['年度'].isin(sel_yrs)) & (df_all['國家'].isin(sel_countries)) & (df_all['市場區域'].isin(sel_area))]

    if not df.empty:
        # KPI 卡片
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("總銷售金額", f"NT${df['金額'].sum():,.0f}")
        k2.metric("收費訂單量", f"{df['收費量'].sum():,.0f}")
        k3.metric("FOC 總量", f"{df['FOC總量'].sum():,.0f}")
        foc_rate = (df['FOC總量'].sum()/(df['數量'].sum()+0.0001)*100)
        k4.metric("平均贈針比", f"{foc_rate:.1f}%")
        k5.metric("活躍國家數", f"{df['國家'].nunique()}")

        tabs = st.tabs(["📊 市場佔比與YoY", "🎯 產品排行", "📉 FOC 專項", "⚡ 營運效率", "🔍 明細查詢"])

        with tabs[0]: # 市場成長與佔比
            yr_str = ", ".join(map(str, sel_yrs))
            st.subheader(f"🌍 全球市場權重分析 ({yr_str})")
            metric_opt = st.selectbox("選擇分析指標", ["銷售金額", "收費訂單數量", "FOC 總數量"])
            m_col = {'銷售金額': '金額', '收費訂單數量': '收費量', 'FOC 總數量': 'FOC總量'}[metric_opt]
            
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.pie(df.groupby('國家')[m_col].sum().reset_index(), values=m_col, names='國家', hole=0.4, title=f"{yr_str} 各國份額", color_discrete_sequence=px.colors.qualitative.Pastel), use_container_width=True)
            
            # YoY 表格
            pivot_df = df.pivot_table(index='國家', columns='年度', values=m_col, aggfunc='sum').fillna(0)
            sorted_cols = sorted(pivot_df.columns)
            for i in range(1, len(sorted_cols)):
                curr, prev = sorted_cols[i], sorted_cols[i-1]
                pivot_df[f"{curr} YoY%"] = ((pivot_df[curr] - pivot_df[prev]) / (pivot_df[prev] + 0.001) * 100).replace([np.inf, -np.inf], 0)
            
            ordered = []
            for y in sorted_cols:
                ordered.append(y)
                if f"{y} YoY%" in pivot_df.columns: ordered.append(f"{y} YoY%")
            
            def style_growth(val, col):
                return f'color: {"green" if val > 0 else "red" if val < 0 else "black"}; font-weight: bold' if 'YoY%' in str(col) else ''
            st.dataframe(pivot_df[ordered].style.format("{:,.1f}").apply(lambda x: [style_growth(v, x.name) for v in x], axis=0))

        with tabs[1]: # 產品分析
            p_df = df.groupby('產品')['金額'].sum().sort_values(ascending=False).reset_index().head(12)
            st.plotly_chart(px.bar(p_df, x='金額', y='產品', orientation='h', color='金額', title="全球產品銷售排行榜"), use_container_width=True)

        with tabs[2]: # FOC
            f_cols = ['培訓與實操用針', '醫師酬勞針', '市場贊助與樣品', '客訴補償']
            f_sum = df[f_cols].sum().reset_index()
            f_sum.columns = ['類別', '數量']
            st.plotly_chart(px.bar(f_sum, x='類別', y='數量', color='類別', text_auto=True, title="FOC 原因分佈"), use_container_width=True)
            st.dataframe(df[df['FOC總量']>0][['銷貨日期', '國家', '客戶', '產品', '數量', '備註']], use_container_width=True)

        with tabs[3]: # 營運效率
            ec1, ec2 = st.columns(2)
            eff_df = df.groupby('國家').agg({'收費量':'sum', 'FOC總量':'sum'}).reset_index()
            eff_df['轉換率'] = eff_df['收費量'] / (eff_df['FOC總量'] + 0.001)
            ec1.plotly_chart(px.bar(eff_df.sort_values('轉換率', ascending=False), x='國家', y='轉換率', title="FOC 投資轉換率 (每支贈針換回訂單量)", color='轉換率'), use_container_width=True)
            
            area_rev = df.groupby('市場區域')['金額'].sum().reset_index()
            ec2.plotly_chart(px.pie(area_rev, values='金額', names='市場區域', title="全球區域營收佔比", hole=0.5), use_container_width=True)

        with tabs[4]: # 數據明細
            search_q = st.text_input("搜尋客戶或產品...").lower()
            q_df = df.copy()
            if search_q:
                q_df = q_df[q_df['客戶'].str.lower().str.contains(search_q, na=False) | q_df['產品'].str.lower().str.contains(search_q, na=False)]
            st.dataframe(q_df[['銷貨日期', '市場區域', '國家', '客戶', '產品', '數量', '金額', '備註']], use_container_width=True)
    else:
        st.warning("⚠️ 此篩選條件下無數據。")
else:
    st.info("👋 管理員您好，請上傳包含「新/舊系統」工作表的 Excel 檔案。系統會自動進行跨系統整合與連動分析。")
