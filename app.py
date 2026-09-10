import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# --- 1. 頁面配置 (精密淨白風格) ---
st.set_page_config(page_title="双美全球銷售數據系統", layout="wide")
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
    
    # 抓取 BX 欄位 (第 76 欄) 作為優先備註
    bx_note_series = None
    if df.shape[1] >= 76:
        bx_note_series = df.iloc[:, 75].astype(str).replace('nan', '')
    
    # 清理欄位名稱
    df.columns = [str(c).replace(' ', '').replace('\n', '').upper() for c in df.columns]
    
    mapping = {
        'DATE': ['銷貨日期', '單據日期', '日期', 'DATE', '銷貨日期A'],
        'COUNTRY': ['國家', '地區', 'COUNTRY', '區域', '廠別名稱'],
        'CUSTOMER': ['客戶簡稱', '客戶', '客戶名稱', 'CUSTOMER', '送貨客戶全名'],
        'PRODUCT': ['品名', '業務品名', '產品名稱', 'PRODUCT', '品號', '業務規格'],
        'QTY': ['銷貨數量', '數量', '計價數量', '總數量', 'QTY'],
        'GIFT_QTY': ['贈/備品量', '贈品量', '贈品數量', 'GIFTQTY'],
        'PRICE': ['單價', '單價NT', 'PRICE', '單   價'],
        'AMOUNT': ['本幣未稅金額', '本幣合計', '金額', '未稅金額', 'AMOUNT', '總計'],
        'NOTE': ['備註', '備註1', 'REMARK', '說明', '備    註']
    }

    def find_best_col(targets):
        for t in targets:
            if t.upper() in df.columns: return t.upper()
        return None

    def safe_get_series(key, is_num=False):
        col = find_best_col(mapping[key])
        if col is not None:
            if is_num:
                s = df[col].astype(str).str.replace(',', '').str.strip()
                return pd.to_numeric(s, errors='coerce').fillna(0)
            return df[col]
        return pd.Series([0] * len(df)) if is_num else pd.Series([""] * len(df))

    p_df = pd.DataFrame()
    
    # 1. 基本欄位處理
    c_date_col = find_best_col(mapping['DATE'])
    p_df['銷貨日期'] = pd.to_datetime(df[c_date_col], errors='coerce') if c_date_col else pd.Timestamp.now()
    p_df['年度'] = p_df['銷貨日期'].dt.year.fillna(0).astype(int) # 強制轉整數年份
    p_df['月份'] = p_df['銷貨日期'].dt.month.fillna(0).astype(int)
    
    p_df['國家'] = safe_get_series('COUNTRY').fillna('台灣').astype(str)
    def classify_region(c):
        c_str = str(c).strip()
        if c_str in ['台灣', '臺灣', 'TAIWAN']: return '台灣市場'
        if c_str in ['大陸', '中國', 'CHINA', '中 國']: return '中國市場'
        return '海外市場'
    p_df['市場區域'] = p_df['國家'].apply(classify_region)
    
    p_df['客戶'] = safe_get_series('CUSTOMER').fillna('未知客戶').astype(str)
    p_df['產品'] = safe_get_series('PRODUCT').astype(str).str.replace('Sunmax DeusaDerm', 'Sunmax Deusaderm VITAL', case=False)
    
    # 2. 數值與 FOC
    p_df['數量'] = safe_get_series('QTY', True)
    p_df['贈品量_單獨欄位'] = safe_get_series('GIFT_QTY', True)
    p_df['單價'] = safe_get_series('PRICE', True)
    p_df['金額'] = safe_get_series('AMOUNT', True)
    
    if bx_note_series is not None and bx_note_series.astype(str).str.len().sum() > 0:
        p_df['備註'] = bx_note_series.values
    else:
        p_df['備註'] = safe_get_series('NOTE').astype(str).replace('nan', '')

    # FOC 邏輯 (全捕捉版)
    p_df['基礎FOC數量'] = np.where((p_df['單價'] <= 0) | (p_df['金額'] <= 0), p_df['數量'], 0)
    p_df['實際FOC總量'] = p_df['基礎FOC數量'] + p_df['贈品量_單獨欄位']
    
    kw = {
        '培訓與實操用針': 'Workshop|實操|示範|培訓|講義|Demo|施打|教學|手法',
        '醫師酬勞針': '酬勞|講師|醫師|陳咸伸|王柏鈞|技術費|DR',
        '市場贊助與樣品': 'Sponsorship|Window|Sample|樣品|贊助|Influencer|廣告|活動',
        '客訴補償': 'complaints|客訴|補償|瑕疵|更換|due to customer|品質'
    }
    for label in kw.keys(): p_df[label] = 0
    p_df['其他待分類FOC'] = 0

    def do_classify(row):
        f_qty = row['實際FOC總量']
        if f_qty <= 0: return row
        matched = False
        rem = str(row['備註'])
        for label, pat in kw.items():
            import re
            if re.search(pat, rem, re.IGNORECASE):
                row[label] = f_qty
                matched = True
                break
        if not matched: row['其他待分類FOC'] = f_qty
        return row
    p_df = p_df.apply(do_classify, axis=1)
    
    p_df['FOC總量'] = p_df[list(kw.keys()) + ['其他待分類FOC']].sum(axis=1)
    p_df['收費量'] = np.where(p_df['金額'] > 0, p_df['數量'], 0)
    
    return p_df

# --- 3. 側邊欄 ---
with st.sidebar:
    st.title("⚙️ 數據導入")
    admin_mode = st.toggle("管理員模式")
    uploaded_file = None
    if admin_mode:
        pwd = st.text_input("管理密碼", type="password")
        if pwd == "sunmax888":
            uploaded_file = st.file_uploader("上傳 Excel", type=["xlsx"])

# --- 4. 主畫面 ---
st.title("📊 双美全球銷售數據")

if uploaded_file:
    all_sheets = pd.read_excel(uploaded_file, sheet_name=None)
    all_data_list = [smart_normalize(df_temp) for name, df_temp in all_sheets.items() if not df_temp.empty]
    df_all = pd.concat(all_data_list, ignore_index=True)
    
    # 頂部篩選
    sc1, sc2, sc3 = st.columns([1, 1, 2])
    sel_area = sc1.multiselect("🏙️ 區域", options=['台灣市場', '中國市場', '海外市場'], default=['海外市場', '中國市場'])
    filtered_by_area = df_all[df_all['市場區域'].isin(sel_area)]
    available_countries = sorted(filtered_by_area['國家'].unique())
    sel_countries = sc3.multiselect("📍 國家", options=available_countries, default=available_countries)
    
    # 年度處理 (轉整數避免 .0)
    yrs = sorted([int(y) for y in df_all['年度'].dropna().unique() if y > 0])
    sel_yrs = sc2.multiselect("📅 年度", yrs, default=yrs[-2:] if len(yrs)>1 else yrs)

    df = df_all[(df_all['年度'].isin(sel_yrs)) & (df_all['國家'].isin(sel_countries)) & (df_all['市場區域'].isin(sel_area))]

    if not df.empty:
        yr_str = ", ".join([f"{y}年" for y in sorted(sel_yrs)])
        st.write(f"🔍 **目前數據統計區間：{yr_str}**")

        # KPI
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("總銷售金額", f"NT${df['金額'].sum():,.0f}")
        k2.metric("收費訂單量", f"{df['收費量'].sum():,.0f}")
        k3.metric("FOC 總量", f"{df['FOC總量'].sum():,.0f}")
        total_real_qty = df['收費量'].sum() + df['FOC總量'].sum()
        k4.metric("平均贈針比", f"{(df['FOC總量'].sum() / (total_real_qty + 0.0001) * 100):.1f}%")
        k5.metric("活躍國家數", f"{df['國家'].nunique()}")

        tabs = st.tabs(["📊 市場佔比與YoY", "🎯 產品排行", "📉 FOC 專項分析", "⚡ 營運效率", "🔍 明細查詢"])

        with tabs[0]: # 市場成長與佔比
            metric_opt = st.selectbox("選擇分析指標", ["銷售金額", "收費訂單數量", "FOC 總數量"])
            m_col = {'銷售金額': '金額', '收費訂單數量': '收費量', 'FOC 總數量': 'FOC總量'}[metric_opt]
            
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.pie(df.groupby('國家')[m_col].sum().reset_index(), values=m_col, names='國家', hole=0.4, title=f"各國佔比 ({yr_str})", color_discrete_sequence=px.colors.qualitative.Pastel), use_container_width=True)
            
            # --- 關鍵修正：格式化年份標題 ---
            pivot_df = df.pivot_table(index='國家', columns='年度', values=m_col, aggfunc='sum').fillna(0)
            
            # 計算 YoY
            display_df = pivot_df.copy()
            sorted_years = sorted(pivot_df.columns)
            for i in range(1, len(sorted_years)):
                curr, prev = sorted_years[i], sorted_years[i-1]
                display_df[f"{curr}年 成長率%"] = ((pivot_df[curr] - pivot_df[prev]) / (pivot_df[prev] + 0.001) * 100).replace([np.inf, -np.inf], 0)
            
            # 重新排列並重新命名標題 (加上 "年" 字)
            final_cols = []
            for y in sorted_years:
                final_cols.append(y)
                growth_col = f"{y}年 成長率%"
                if growth_col in display_df.columns: final_cols.append(growth_col)
            
            display_df = display_df[final_cols]
            # 將原本是數字的標題 (例如 2026) 轉成 "2026年"
            display_df.columns = [f"{int(c)}年" if isinstance(c, (int, float)) else c for c in display_df.columns]

            def style_growth(val, col):
                if '成長率' in str(col):
                    return f'color: {"green" if val > 0 else "red" if val < 0 else "black"}; font-weight: bold'
                return ''
            
            st.write(f"#### 逐年趨勢細節")
            st.dataframe(display_df.style.format("{:,.1f}").apply(lambda x: [style_growth(v, x.name) for v in x], axis=0))

        with tabs[1]: # 產品分析
            p_df = df.groupby('產品')['金額'].sum().sort_values(ascending=False).reset_index().head(12)
            st.plotly_chart(px.bar(p_df, x='金額', y='產品', orientation='h', color='金額', title="全球產品銷售排行榜"), use_container_width=True)

        with tabs[2]: # FOC 專項
            f_cols = ['培訓與實操用針', '醫師酬勞針', '市場贊助與樣品', '客訴補償', '其他待分類FOC']
            f_sum = df[f_cols].sum().reset_index()
            f_sum.columns = ['類別', '數量']
            st.plotly_chart(px.bar(f_sum, x='類別', y='數量', color='類別', text_auto=True), use_container_width=True)
            st.dataframe(df[df['FOC總量']>0][['銷貨日期', '國家', '客戶', '產品', '數量', '贈品量_單單獨欄位', '金額', '備註']], use_container_width=True)

        with tabs[4]: # 明細
            search_q = st.text_input("搜尋客戶或產品...").lower()
            q_df = df.copy()
            if search_q:
                q_df = q_df[q_df['客戶'].str.lower().str.contains(search_q, na=False) | q_df['產品'].str.lower().str.contains(search_q, na=False)]
            # 顯示時也把年份格式化
            q_df['年度'] = q_df['年度'].astype(str) + "年"
            st.dataframe(q_df[['銷貨日期', '市場區域', '國家', '客戶', '產品', '數量', '金額', '備註']], use_container_width=True)
    else:
        st.warning("⚠️ 此篩選條件下無數據。")
else:
    st.info("👋 管理員您好，請導入 Excel 檔案。系統會自動格式化年份並分析數據。")
