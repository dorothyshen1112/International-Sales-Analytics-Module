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
    [data-testid="stMetricValue"] { font-size: 1.6rem !important; color: #0984E3; }
    h1, h2, h3 { color: #0984E3; font-family: 'Helvetica Neue', sans-serif; }
    div.stTabs [data-baseweb="tab-list"] { gap: 15px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 智慧欄位匹配引擎 (核心：處理新舊系統差異) ---
def smart_normalize(df):
    # 保存原始 BX 欄位 (新系統專用：第 76 欄，索引 75)
    bx_note = None
    if df.shape[1] >= 76:
        bx_note = df.iloc[:, 75].astype(str).replace('nan', '')
    
    # 清理欄位名稱 (去空格、轉大寫)
    df.columns = [str(c).replace(' ', '').replace('\n', '').upper() for c in df.columns]
    
    # 定義新舊系統可能的欄位名稱對應
    mapping = {
        'DATE': ['銷貨日期', '單據日期', '日期', 'DATE', '銷貨日期A', '單據日'],
        'COUNTRY': ['國家', '地區', 'COUNTRY', '區域', '廠別名稱'],
        'CUSTOMER': ['客戶簡稱', '客戶', '客戶名稱', 'CUSTOMER', '送貨客戶全名'],
        'PRODUCT': ['品名', '業務品名', '產品名稱', 'PRODUCT', '業務規格', '品號'],
        'QTY': ['銷貨數量', '數量', '計價數量', '總數量', 'QTY', '件數'],
        'PRICE': ['單價', '單價NT', 'PRICE'],
        'AMOUNT': ['本幣未稅金額', '本幣合計', '金額', '未稅金額', 'AMOUNT', '總計', '本幣未稅金額_1'],
        'NOTE': ['備註', '備註1', 'REMARK', '說明', '其他']
    }

    def find_best(targets):
        for t in targets:
            if t.upper() in df.columns: return t.upper()
        return None

    # 數據正規化提取至新 DataFrame
    p_df = pd.DataFrame()
    
    c_date = find_best(mapping['DATE'])
    p_df['銷貨日期'] = pd.to_datetime(df[c_date], errors='coerce') if c_date else pd.Timestamp.now()
    
    c_country = find_best(mapping['COUNTRY'])
    p_df['國家'] = df[c_country].fillna('台灣') if c_country else '台灣'
    
    c_cust = find_best(mapping['CUSTOMER'])
    p_df['客戶'] = df[c_cust].fillna('未知客戶') if c_cust else '未知客戶'
    
    c_prod = find_best(mapping['PRODUCT'])
    # 產品合併邏輯：統一更名
    p_df['產品'] = df[c_prod].astype(str).replace('Sunmax DeusaDerm', 'Sunmax Deusaderm VITAL') if c_prod else "未知產品"
    
    c_qty = find_best(mapping['QTY'])
    p_df['數量'] = pd.to_numeric(df[c_qty].astype(str).str.replace(',', ''), errors='coerce').fillna(0) if c_qty else 0
    
    c_price = find_best(mapping['PRICE'])
    p_df['單價'] = pd.to_numeric(df[c_price].astype(str).str.replace(',', ''), errors='coerce').fillna(0) if c_price else 0

    c_amt = find_best(mapping['AMOUNT'])
    p_df['金額'] = pd.to_numeric(df[c_amt].astype(str).str.replace(',', ''), errors='coerce').fillna(0) if c_amt else 0
    
    # 備註處理：如果是新系統有 BX 欄位就用 BX，否則找 NOTE
    if bx_note is not None and len(bx_note.unique()) > 1: # 簡單判斷 BX 是否有意義
        p_df['備註'] = bx_note
    else:
        c_note = find_best(mapping['NOTE'])
        p_df['備註'] = df[c_note].astype(str).replace('nan', '') if c_note else ""

    # 計算基本屬性
    p_df['年度'] = p_df['銷貨日期'].dt.year
    p_df['月份'] = p_df['銷貨日期'].dt.month
    p_df['市場區域'] = p_df['國家'].apply(lambda x: '台灣市場' if '台灣' in str(x) else '海外市場')
    
    # FOC 四大類別自動分類
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
    st.title("⚙️ 系統管理")
    admin_mode = st.toggle("管理員模式 (導入雙系統 Excel)")
    uploaded_file = None
    if admin_mode:
        pwd = st.text_input("管理密碼", type="password")
        if pwd == "sunmax888":
            uploaded_file = st.file_uploader("上傳包含多個 Sheet 的 Excel", type=["xlsx"])

st.title("🌐 雙美全球銷售數據指揮中心")

if uploaded_file:
    # 智慧讀取：讀取所有工作表 (Sheet)
    all_sheets = pd.read_excel(uploaded_file, sheet_name=None)
    
    all_data_list = []
    for sheet_name, df_temp in all_sheets.items():
        if not df_temp.empty:
            # 對每個 Sheet 進行欄位正規化
            processed_sheet = smart_normalize(df_temp)
            if not processed_sheet.empty:
                all_data_list.append(processed_sheet)
    
    # 垂直合併所有 Sheet 數據
    df_all = pd.concat(all_data_list, ignore_index=True)
    
    # 全域篩選器
    st.info(f"✅ 已辨識並合併 {len(all_data_list)} 個工作表數據。")
    fc1, fc2, fc3 = st.columns([1, 1, 2])
    sel_area = fc1.multiselect("🏙️ 市場區域", ['台灣市場', '海外市場'], default=['海外市場'])
    
    yrs = sorted(df_all['年度'].dropna().unique().astype(int))
    sel_yrs = fc2.multiselect("📅 分析年度", yrs, default=yrs[-2:] if len(yrs)>1 else yrs)
    
    countries = sorted(df_all[df_all['市場區域'].isin(sel_area)]['國家'].unique())
    sel_countries = fc3.multiselect("📍 分析國家", countries, default=countries)

    # 應用篩選
    df = df_all[(df_all['年度'].isin(sel_yrs)) & (df_all['國家'].isin(sel_countries)) & (df_all['市場區域'].isin(sel_area))]

    if not df.empty:
        # KPI 頂部摘要
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("所選區域營收", f"NT${df['金額'].sum():,.0f}")
        k2.metric("收費訂單量", f"{df['收費量'].sum():,.0f}")
        k3.metric("FOC 總量", f"{df['FOC總量'].sum():,.0f}")
        k4.metric("贈針比 (FOC %)", f"{(df['FOC總量'].sum()/(df['數量'].sum()+0.001)*100):.1%}%")
        k5.metric("國家數", f"{df['國家'].nunique()}")

        tabs = st.tabs(["📊 佔比與 YoY 成長", "🎯 產品分析", "📉 FOC 專項", "⚡ 營運效率(進階)", "🔍 數據明細"])

        with tabs[0]: # 市場佔比與成長
            metric_opt = st.selectbox("選擇分析指標", ["銷售金額", "收費訂單數量", "FOC 總數量"])
            m_col = {'銷售金額': '金額', '收費訂單數量': '收費量', 'FOC 總數量': 'FOC總量'}[metric_opt]
            
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.pie(df.groupby('國家')[m_col].sum().reset_index(), values=m_col, names='國家', hole=0.4, title=f"各國 {metric_opt} 佔比", color_discrete_sequence=px.colors.qualitative.Pastel), use_container_width=True)
            
            # 逐年成長表格
            pivot_df = df.pivot_table(index='國家', columns='年度', values=m_col, aggfunc='sum').fillna(0)
            for i in range(1, len(pivot_df.columns)):
                curr, prev = pivot_df.columns[i], pivot_df.columns[i-1]
                pivot_df[f"{curr} YoY%"] = ((pivot_df[curr] - pivot_df[prev]) / (pivot_df[prev] + 0.001) * 100).replace([np.inf, -np.inf], 0)
            st.write("#### 逐年成長明細 (YoY%)")
            st.dataframe(pivot_df.style.format("{:,.1f}"))

        with tabs[1]: # 產品分析
            st.subheader("🎯 全球產品銷售排行榜")
            p_df = df.groupby('產品')['金額'].sum().sort_values(ascending=False).reset_index().head(12)
            st.plotly_chart(px.bar(p_df, x='金額', y='產品', orientation='h', color='金額', color_continuous_scale='Blues'), use_container_width=True)

        with tabs[2]: # FOC
            st.subheader("📉 FOC 資源配置原因分析")
            f_cols = ['培訓與實操用針', '醫師酬勞針', '市場贊助與樣品', '客訴補償']
            f_sum = df[f_cols].sum().reset_index()
            f_sum.columns = ['類別', '數量']
            st.plotly_chart(px.bar(f_sum, x='類別', y='數量', color='類別', text_auto=True), use_container_width=True)
            st.dataframe(df[df['FOC總量']>0][['銷貨日期', '國家', '產品', '數量', '備註']], use_container_width=True)

        with tabs[3]: # 營運效率
            st.subheader("⚡ 投資回報與物流效率")
            ec1, ec2 = st.columns(2)
            eff_df = df.groupby('國家').agg({'收費量':'sum', 'FOC總量':'sum'}).reset_index()
            eff_df['轉換率'] = eff_df['收費量'] / (eff_df['FOC總量'] + 0.001)
            ec1.plotly_chart(px.bar(eff_df.sort_values('轉換率', ascending=False), x='國家', y='轉換率', title="FOC 投資轉換率 (1支贈針換X支收費)", color='轉換率'), use_container_width=True)
            
            logi_df = df[df['收費量']>0].groupby('國家').agg({'數量':'sum', '銷貨日期':'count'}).reset_index()
            logi_df['單筆採購量'] = logi_df['數量'] / logi_df['銷貨日期']
            ec2.plotly_chart(px.bar(logi_df.sort_values('單筆採購量', ascending=False), x='國家', y='單筆採購量', title="物流效率 (平均單筆訂單件數)", color='單筆採購量', color_continuous_scale='Greens'), use_container_width=True)

        with tabs[4]: # 數據明細
            st.subheader("🔍 全球銷售明細查詢")
            search_q = st.text_input("輸入客戶或品名關鍵字搜尋...").lower()
            q_df = df.copy()
            if search_q:
                q_df = q_df[q_df['客戶'].str.lower().str.contains(search_q, na=False) | q_df['產品'].str.lower().str.contains(search_q, na=False)]
            st.dataframe(q_df[['銷貨日期', '市場區域', '國家', '客戶', '產品', '數量', '金額', '備註']], use_container_width=True)

else:
    st.info("👋 管理員您好，請上傳包含「新系統」與「舊系統」Sheet 的 Excel 檔案。系統會自動進行跨系統整合分析。")
