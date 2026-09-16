import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import re

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
    
    # 抓取 BX 欄位 (第 76 欄) 作為優先備註來源
    bx_note_series = None
    if df.shape[1] >= 76:
        bx_note_series = df.iloc[:, 75].astype(str).replace('nan', '')
    
    df.columns = [str(c).replace(' ', '').replace('\n', '').upper() for c in df.columns]
    
    mapping = {
        'DATE': ['銷貨日期', '單據日期', '日期', 'DATE', '銷貨日期A'],
        'COUNTRY': ['國家', '地區', 'COUNTRY', '區域', '廠別名稱'],
        'CUSTOMER': ['客戶簡稱', '客戶', '客戶名稱', 'CUSTOMER', '送貨客戶全名'],
        'PRODUCT': ['品名', '業務品名', '產品名稱', 'PRODUCT', '品號', '業務規格'],
        'QTY': ['銷貨數量', '數量', '計價數量', '總數量', 'QTY'],
        'GIFT_QTY': ['贈/備品量', '贈品量', '贈品數量', 'GIFTQTY', '贈/備品數量'],
        'PRICE': ['單價', '單價NT', 'PRICE'],
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
    c_date_col = find_best_col(mapping['DATE'])
    p_df['銷貨日期'] = pd.to_datetime(df[c_date_col], errors='coerce') if c_date_col else pd.Timestamp.now()
    p_df['年度'] = p_df['銷貨日期'].dt.year.fillna(0).astype(int)
    p_df['國家'] = safe_get_series('COUNTRY').fillna('台灣').astype(str)
    
    def classify_region(c):
        c_str = str(c).strip()
        if c_str in ['台灣', '臺灣', 'TAIWAN']: return '台灣市場'
        if c_str in ['大陸', '中國', 'CHINA', '中 國', '大 陸']: return '中國市場'
        return '海外市場'
    p_df['市場區域'] = p_df['國家'].apply(classify_region)
    
    p_df['客戶'] = safe_get_series('CUSTOMER').fillna('未知客戶').astype(str)
    # 修正產品名稱重複 VITAL 的問題
    raw_prod = safe_get_series('PRODUCT').astype(str)
    p_df['產品'] = raw_prod.str.replace('Sunmax DeusaDerm', 'Sunmax Deusaderm VITAL', case=False).str.replace('VITAL VITAL', 'VITAL', case=False)
    
    p_df['數量'] = safe_get_series('QTY', True)
    p_df['贈品量_單獨欄位'] = safe_get_series('GIFT_QTY', True)
    p_df['單價'] = safe_get_series('PRICE', True)
    p_df['金額'] = safe_get_series('AMOUNT', True)
    
    if bx_note_series is not None and bx_note_series.astype(str).str.len().sum() > 0:
        p_df['備註'] = bx_note_series.values
    else:
        p_df['備註'] = safe_get_series('NOTE').astype(str).replace('nan', '')

    # --- FOC 核心邏輯 (全量捕捉 + 標籤分類) ---
    p_df['實際FOC數量'] = np.where((p_df['單價'] <= 0) | (p_df['金額'] <= 0), p_df['數量'], 0) + p_df['贈品量_單獨欄位']
    
    kw = {
        'Workshop Training': 'Workshop|實操|示範|培訓|教學|手法|Demo',
        'Rebate': 'Rebate|回饋|返利|折扣',
        '醫師酬勞': '酬勞|講師|醫師|陳咸伸|王柏鈞|技術費|DR',
        '市場贊助與樣品': 'Sponsorship|Window|Sample|樣品|贊助|Influencer|廣告|活動',
        '客訴補償': 'complaints|客訴|補償|瑕疵|更換|quality'
    }
    
    p_df['FOC類別'] = '非FOC'
    for label in kw.keys(): p_df[label] = 0
    p_df['其他待分類FOC'] = 0

    def do_classify(row):
        f_qty = row['實際FOC數量']
        if f_qty <= 0: return row
        
        matched_label = None
        rem = str(row['備註'])
        for label, pat in kw.items():
            if re.search(pat, rem, re.IGNORECASE):
                row[label] = f_qty
                matched_label = label
                break
        
        if matched_label:
            row['FOC類別'] = matched_label
        else:
            row['其他待分類FOC'] = f_qty
            row['FOC類別'] = '其他待分類FOC'
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
    available_yrs = sorted([int(y) for y in df_all['年度'].dropna().unique() if y > 0])
    sel_yrs = sc2.multiselect("📅 年度", available_yrs, default=available_yrs)

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
        k4.metric("平均贈針比", f"{(df['FOC總量'].sum() / (total_real_qty + 0.0001) * 100):.1%}")
        k5.metric("活躍國家數", f"{df['國家'].nunique()}")

        tabs = st.tabs(["📊 市場佔比與YoY", "🎯 產品排行", "📉 FOC 專項分析", "⚡ 營運效率", "🔍 明細查詢"])

        with tabs[2]: # FOC 專項分析
            st.subheader(f"📉 FOC 原因分佈 ({yr_str})")
            f_cols = ['Workshop Training', 'Rebate', '醫師酬勞', '市場贊助與樣品', '客訴補償', '其他待分類FOC']
            f_sum = df[f_cols].sum().reset_index()
            f_sum.columns = ['類別', '數量']
            st.plotly_chart(px.bar(f_sum, x='類別', y='數量', color='類別', text_auto=True), use_container_width=True)
            
            st.divider()
            st.write("#### 🔍 FOC 分類核對工具")
            # --- 關鍵功能：新增類別過濾器 ---
            target_foc_cat = st.selectbox("選擇要查看的區塊 (類別)：", options=["全部 FOC"] + f_cols)
            
            foc_display_df = df[df['FOC總量'] > 0].copy()
            if target_foc_cat != "全部 FOC":
                foc_display_df = foc_display_df[foc_display_df['FOC類別'] == target_foc_cat]
            
            st.write(f"顯示類別：**{target_foc_cat}** (共 {len(foc_display_df)} 筆資料)")
            st.dataframe(foc_display_df[['銷貨日期', '國家', '客戶', '產品', '數量', '贈品量_單獨欄位', '金額', '備註']], use_container_width=True)

        with tabs[0]: # YoY 成長
            metric_opt = st.selectbox("選擇指標", ["銷售金額", "收費訂單數量", "FOC 總數量"])
            m_col = {'銷售金額': '金額', '收費訂單數量': '收費量', 'FOC 總數量': 'FOC總量'}[metric_opt]
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.pie(df.groupby('國家')[m_col].sum().reset_index(), values=m_col, names='國家', hole=0.4, title=f"各國份額 ({yr_str})", color_discrete_sequence=px.colors.qualitative.Pastel), use_container_width=True)
            pivot_df = df.pivot_table(index='國家', columns='年度', values=m_col, aggfunc='sum').fillna(0)
            display_df = pivot_df.copy()
            sorted_years = sorted(pivot_df.columns)
            for i in range(1, len(sorted_years)):
                curr, prev = sorted_years[i], sorted_years[i-1]
                display_df[f"{curr}年 成長率%"] = np.where(pivot_df[prev] == 0, np.nan, ((pivot_df[curr] - pivot_df[prev]) / pivot_df[prev] * 100))
            ordered = []
            for y in sorted_years:
                ordered.append(y)
                if f"{y}年 成長率%" in display_df.columns: ordered.append(f"{y}年 成長率%")
            display_pivot = display_df[ordered].copy()
            display_pivot.columns = [f"{int(c)}年" if isinstance(c, (int, float)) else c for c in display_pivot.columns]
            def format_value(val, col_name):
                if '成長率%' in str(col_name):
                    if pd.isna(val): return "-"
                    return f"{val:.1f}%"
                return f"{val:,.0f}"
            def style_growth(val, col_name):
                if '成長率%' in str(col_name):
                    if pd.isna(val): return 'color: gray'
                    if val > 0: return 'color: red; font-weight: bold'
                    if val < 0: return 'color: green; font-weight: bold'
                return ''
            formatted_df = display_pivot.copy().astype(object)
            for col in display_pivot.columns:
                formatted_df[col] = display_pivot[col].apply(lambda x: format_value(x, col))
            st.dataframe(formatted_df.style.apply(lambda x: [style_growth(display_pivot.loc[x.name, col], col) for col in display_pivot.columns], axis=1))

        with tabs[1]:
            p_df = df.groupby('產品')['金額'].sum().sort_values(ascending=False).reset_index().head(12)
            st.plotly_chart(px.bar(p_df, x='金額', y='產品', orientation='h', color='金額', title="全球產品銷售排行榜"), use_container_width=True)
        with tabs[3]:
            eff_df = df.groupby('國家').agg({'收費量':'sum', 'FOC總量':'sum'}).reset_index()
            eff_df['轉換率'] = (eff_df['收費量'] / (eff_df['FOC總量'] + 0.001)).round(1)
            st.plotly_chart(px.bar(eff_df.sort_values('轉換率', ascending=False), x='國家', y='轉換率', title="FOC 投資轉換率", color='轉換率', text_auto=True), use_container_width=True)
        with tabs[4]:
            search_q = st.text_input("搜尋客戶或產品...").lower()
            q_df = df.copy()
            if search_q:
                q_df = q_df[q_df['客戶'].str.lower().str.contains(search_q, na=False) | q_df['產品'].str.lower().str.contains(search_q, na=False)]
            st.dataframe(q_df[['銷貨日期', '市場區域', '國家', '客戶', '產品', '數量', '金額', '備註']], use_container_width=True)
else:
    st.info("👋 管理員您好，請導入 Excel。本版本已加入 FOC 類別核對工具與產品名稱修正。")
