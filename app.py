import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# --- 1. 頁面配置 (精密淨白風格) ---
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

# --- 2. 智慧資料正規化引擎 ---
def smart_normalize(df):
    if df.empty: return pd.DataFrame()
    
    # 抓取 BX 欄位 (第76欄)
    bx_note = df.iloc[:, 75].astype(str).replace('nan', '') if df.shape[1] >= 76 else None
    
    # 清理欄位名稱
    df.columns = [str(c).replace(' ', '').replace('\n', '').upper() for c in df.columns]
    
    mapping = {
        'DATE': ['銷貨日期', '單據日期', '日期', 'DATE', '銷貨日期A'],
        'COUNTRY': ['國家', '地區', 'COUNTRY', '區域', '廠別名稱'],
        'CUSTOMER': ['客戶簡稱', '客戶', '客戶名稱', 'CUSTOMER'],
        'PRODUCT': ['品名', '業務品名', '產品名稱', 'PRODUCT', '品號'],
        'QTY': ['銷貨數量', '數量', '計價數量', '總數量', 'QTY'],
        'PRICE': ['單價', '單價NT', 'PRICE'],
        'AMOUNT': ['本幣未稅金額', '本幣合計', '金額', '未稅金額', 'AMOUNT', '總計'],
        'NOTE': ['備註', '備註1', 'REMARK', '說明']
    }

    def find_best(targets):
        for t in targets:
            if t.upper() in df.columns: return t.upper()
        return None

    p_df = pd.DataFrame()
    c_date = find_best(mapping['DATE'])
    p_df['銷貨日期'] = pd.to_datetime(df[c_date], errors='coerce') if c_date else pd.Timestamp.now()
    
    c_country = find_best(mapping['COUNTRY'])
    p_df['國家'] = df[c_country].fillna('台灣').astype(str) if c_country else '台灣'
    
    # --- 需求 2：區域劃分邏輯 (台灣/中國/海外) ---
    def classify_region(c):
        c = c.strip()
        if c in ['台灣', '臺灣', 'TAIWAN']: return '台灣市場'
        if c in ['大陸', '中國', 'CHINA', '中 國']: return '中國市場'
        return '海外市場'
    
    p_df['市場區域'] = p_df['國家'].apply(classify_region)
    
    c_cust = find_best(mapping['CUSTOMER'])
    p_df['客戶'] = df[c_cust].fillna('未知客戶') if c_cust else '未知客戶'
    
    c_prod = find_best(mapping['PRODUCT'])
    p_df['產品'] = df[c_prod].astype(str).str.replace('Sunmax DeusaDerm', 'Sunmax Deusaderm VITAL', case=False) if c_prod else "未知產品"
    
    c_qty = find_best(mapping['QTY'])
    p_df['數量'] = pd.to_numeric(df[c_qty].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    
    c_price = find_best(mapping['PRICE'])
    p_df['單價'] = pd.to_numeric(df[c_price].astype(str).str.replace(',', ''), errors='coerce').fillna(0)

    c_amt = find_best(mapping['AMOUNT'])
    p_df['金額'] = pd.to_numeric(df[c_amt].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    
    if bx_note is not None and bx_note.str.len().sum() > 0:
        p_df['備註'] = bx_note
    else:
        c_note = find_best(mapping['NOTE'])
        p_df['備註'] = df[c_note].astype(str).replace('nan', '') if c_note else ""

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
    all_sheets = pd.read_excel(uploaded_file, sheet_name=None)
    all_data_list = [smart_normalize(df_temp) for name, df_temp in all_sheets.items() if not df_temp.empty]
    df_all = pd.concat(all_data_list, ignore_index=True)
    
    # --- 需求 1：修正篩選連動 BUG ---
    st.info("💡 系統已自動整合跨系統數據。請設定下方條件：")
    sc1, sc2, sc3 = st.columns([1, 1, 2])
    
    # A. 先選區域 (台灣/中國/海外)
    area_options = ['台灣市場', '中國市場', '海外市場']
    sel_area = sc1.multiselect("區域", options=area_options, default=['海外市場', '中國市場'])
    
    # B. 根據選定的「區域」，動態過濾「國家」清單
    filtered_by_area = df_all[df_all['市場區域'].isin(sel_area)]
    available_countries = sorted(filtered_by_area['國家'].unique())
    # 國家選單會隨區域變動
    sel_countries = sc3.multiselect("國家", options=available_countries, default=available_countries)
    
    # C. 年度選單
    yrs = sorted(df_all['年度'].dropna().unique().astype(int))
    sel_yrs = sc2.multiselect("年度", yrs, default=yrs[-2:] if len(yrs)>1 else yrs)

    # 最終應用篩選
    df = df_all[(df_all['年度'].isin(sel_yrs)) & (df_all['國家'].isin(sel_countries)) & (df_all['市場區域'].isin(sel_area))]

    if not df.empty:
        # KPI 頂部
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("總銷售金額", f"NT${df['金額'].sum():,.0f}")
        k2.metric("收費訂單量", f"{df['收費量'].sum():,.0f}")
        k3.metric("FOC 總量", f"{df['FOC總量'].sum():,.0f}")
        # 修正百分比顯示 Bug
        foc_rate = (df['FOC總量'].sum()/(df['數量'].sum()+0.0001)*100)
        k4.metric("平均贈針比", f"{foc_rate:.1f}%")
        k5.metric("活躍國家數", f"{df['國家'].nunique()}")

        tabs = st.tabs(["📊 市場佔比與YoY", "🎯 產品排行", "📉 FOC 專項", "⚡ 營運效率", "🔍 明細查詢"])

        with tabs[0]: 
            yr_str = ", ".join(map(str, sel_yrs))
            st.subheader(f"🌍 全球市場權重分析 ({yr_str})")
            metric_opt = st.selectbox("選擇分析指標", ["銷售金額", "收費訂單數量", "FOC 總數量"])
            m_col = {'銷售金額': '金額', '收費訂單數量': '收費量', 'FOC 總數量': 'FOC總量'}[metric_opt]
            
            c1, c2 = st.columns(2)
            # 圓餅圖
            fig_pie = px.pie(df.groupby('國家')[m_col].sum().reset_index(), values=m_col, names='國家', hole=0.4, title=f"{yr_str} 各國佔比", color_discrete_sequence=px.colors.qualitative.Pastel)
            c1.plotly_chart(fig_pie, use_container_width=True)
            
            # 成長趨勢表
            pivot_df = df.pivot_table(index='國家', columns='年度', values=m_col, aggfunc='sum').fillna(0)
            display_df = pivot_df.copy()
            sorted_cols = sorted(pivot_df.columns)
            for i in range(1, len(sorted_cols)):
                curr, prev = sorted_cols[i], sorted_cols[i-1]
                display_df[f"{curr} YoY%"] = ((pivot_df[curr] - pivot_df[prev]) / (pivot_df[prev] + 0.001) * 100).replace([np.inf, -np.inf], 0)
            
            ordered = []
            for y in sorted_cols:
                ordered.append(y)
                if f"{y} YoY%" in display_df.columns: ordered.append(f"{y} YoY%")
            
            def style_growth(val, col):
                return f'color: {"green" if val > 0 else "red" if val < 0 else "black"}; font-weight: bold' if 'YoY%' in str(col) else ''
            st.dataframe(display_df[ordered].style.format("{:,.1f}").apply(lambda x: [style_growth(v, x.name) for v in x], axis=0))

        with tabs[1]: # 產品分析
            st.subheader(f"🎯 全球產品銷售排行榜 ({yr_str})")
            p_df = df.groupby('產品')['金額'].sum().sort_values(ascending=False).reset_index().head(12)
            st.plotly_chart(px.bar(p_df, x='金額', y='產品', orientation='h', color='金額', title="熱銷產品 Top 12"), use_container_width=True)

        with tabs[2]: # FOC 專項
            st.subheader(f"📉 FOC 原因分佈 ({yr_str})")
            f_cols = ['培訓與實操用針', '醫師酬勞針', '市場贊助與樣品', '客訴補償']
            f_sum = df[f_cols].sum().reset_index()
            f_sum.columns = ['類別', '數量']
            st.plotly_chart(px.bar(f_sum, x='類別', y='數量', color='類別', text_auto=True), use_container_width=True)
            st.dataframe(df[df['FOC總量']>0][['銷貨日期', '國家', '客戶', '產品', '數量', '備註']], use_container_width=True)

        with tabs[3]: # 營運效率
            st.subheader("⚡ 投資回報與營收結構")
            ec1, ec2 = st.columns(2)
            # 轉換率
            eff_df = df.groupby('國家').agg({'收費量':'sum', 'FOC總量':'sum'}).reset_index()
            eff_df['轉換率'] = eff_df['收費量'] / (eff_df['FOC總量'] + 0.001)
            ec1.plotly_chart(px.bar(eff_df.sort_values('轉換率', ascending=False), x='國家', y='轉換率', title="FOC 轉換效率 (1支贈針換X支收費)", color='轉換率'), use_container_width=True)
            # 區域佔比
            area_rev = df.groupby('市場區域')['金額'].sum().reset_index()
            ec2.plotly_chart(px.pie(area_rev, values='金額', names='市場區域', title="區域營收貢獻度", hole=0.5), use_container_width=True)

        with tabs[4]: # 數據明細
            search_q = st.text_input("輸入關鍵字搜尋...").lower()
            q_df = df.copy()
            if search_q:
                q_df = q_df[q_df['客戶'].str.lower().str.contains(search_q, na=False) | q_df['產品'].str.lower().str.contains(search_q, na=False)]
            st.dataframe(q_df[['銷貨日期', '市場區域', '國家', '客戶', '產品', '數量', '金額', '備註']], use_container_width=True)

else:
    st.info("👋 管理員您好，請上傳 Excel 檔案。系統會自動進行跨系統整合與區域劃分分析。")
