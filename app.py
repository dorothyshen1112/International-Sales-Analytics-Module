import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import re
import os

# --- 1. 頁面配置 (精密淨白風格) ---
st.set_page_config(page_title="双美全球銷售數據系統", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF; color: #2D3436; }
    .stMetric { background-color: #F8F9FA; border-radius: 10px; padding: 15px; border: 1px solid #E9ECEF; }
    [data-testid="stMetricValue"] { font-size: 1.6rem !important; color: #0984E3; font-weight: 700; }
    h1, h2, h3 { color: #0984E3; font-family: 'Helvetica Neue', sans-serif; }
    div.stTabs [data-baseweb="tab-list"] { gap: 15px; border-bottom: 2px solid #F1F3F5; }
    .report-note { 
        background-color: #E3F2FD; 
        padding: 15px; 
        border-left: 6px solid #2196F3; 
        border-radius: 4px; 
        margin-bottom: 15px; 
        font-size: 0.95rem; 
        color: #1565C0;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 智慧數據處理引擎 ---
def smart_normalize(df):
    if df.empty: return pd.DataFrame()
    
    # 抓取 BX 欄位 (第 76 欄) 作為優先備註來源
    bx_note_series = None
    if df.shape[1] >= 76:
        bx_note_series = df.iloc[:, 75].astype(str).replace('nan', '')
    
    # 清理原始欄位名稱
    df.columns = [str(c).replace(' ', '').replace('\n', '').upper() for c in df.columns]
    
    mapping = {
        'DATE': ['銷貨日期', '單據日期', '日期', 'DATE', '銷貨日期A'],
        'COUNTRY': ['國家', '地區', 'COUNTRY', '區域', '廠別名稱'],
        'CUSTOMER': ['客戶簡稱', '客戶', '客戶名稱', 'CUSTOMER', '送貨客戶全名'],
        'PRODUCT': ['品名', '業務品名', '產品名稱', 'PRODUCT', '品號', '業務規格'],
        'QTY': ['銷貨數量', '數量', '計價數量', '總數量', 'QTY'],
        'GIFT_QTY': ['贈/備品量', '贈品量', '贈品數量', 'GIFTQTY', '贈/備品數量'],
        'PRICE': ['單價', '單價NT', 'PRICE'],
        'AMOUNT': ['本幣未稅金額', '本幣合計', '金額', '未稅金額', 'AMOUNT', '總計']
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
    p_df['月份'] = p_df['銷貨日期'].dt.month.fillna(0).astype(int)
    
    # 強制鎖定 2023+
    p_df = p_df[p_df['年度'] >= 2023].copy()
    if p_df.empty: return pd.DataFrame()

    p_df['國家'] = safe_get_series('COUNTRY').loc[p_df.index].fillna('台灣').astype(str)
    
    # 區域與商務模式劃分
    def classify_region_and_model(c):
        c_str = str(c).strip()
        if c_str in ['台灣', '臺灣', 'TAIWAN']: return '台灣市場', '經銷商模式'
        if c_str in ['大陸', '中國', 'CHINA', '中 國', '大 陸']: return '中國市場', '經銷商模式'
        if c_str in ['日本', 'JAPAN']: return '海外市場', '直營診所模式'
        return '海外市場', '經銷商模式'
    
    p_df['市場區域'], p_df['商務模式'] = zip(*p_df['國家'].apply(classify_region_and_model))
    
    p_df['客戶'] = safe_get_series('CUSTOMER').loc[p_df.index].fillna('未知客戶').astype(str)
    # 修正產品名稱
    raw_prod = safe_get_series('PRODUCT').loc[p_df.index].astype(str)
    p_df['產品'] = raw_prod.str.replace('Sunmax DeusaDerm', 'Sunmax Deusaderm VITAL', case=False).str.replace('VITAL VITAL', 'VITAL', case=False)
    
    p_df['數量'] = safe_get_series('QTY', True).loc[p_df.index]
    p_df['贈品量'] = safe_get_series('GIFT_QTY', True).loc[p_df.index]
    p_df['單價'] = safe_get_series('PRICE', True).loc[p_df.index]
    p_df['金額'] = safe_get_series('AMOUNT', True).loc[p_df.index]
    
    p_df['備註'] = bx_note_series.loc[p_df.index].values if bx_note_series is not None else safe_get_series('NOTE').loc[p_df.index].astype(str).replace('nan', '')

    # FOC 七大分類邏輯
    p_df['實際FOC數量'] = np.where((p_df['單價'] <= 0) | (p_df['金額'] <= 0), p_df['數量'], 0) + p_df['贈品量']
    foc_cats = ['培訓活動用針', '醫師酬勞', 'Workshop Training', 'Training/Rebate for JP', '市場贊助', '研究用針', '客訴補償', 'FOC樣品運輸']
    
    def classify_foc(row):
        f_qty = row['實際FOC數量']
        if f_qty <= 0: return '非FOC'
        rem = str(row['備註']).lower()
        if 'workshop training' in rem: return 'Workshop Training'
        if any(x in rem for x in ['實操針', '示範針', '會員實操針', '培訓針', '培訓施打用針', 'demo', '培訓活動施打用針']): return '培訓活動用針'
        if any(x in rem for x in ['酬勞針', '講師針', 'speech']): return '醫師酬勞'
        if any(x in rem for x in ['rebate', 'training']): return 'Training/Rebate for JP'
        if any(x in rem for x in ['sponsorship', 'launch event']): return '市場贊助'
        if any(x in rem for x in ['sample needles', 'irb']): return '研究用針'
        if any(x in rem for x in ['complaints', '客訴', '補償', '瑕疵', '更換', 'quality', 'customer complaints']): return '客訴補償'
        return 'FOC樣品運輸'

    p_df['FOC類別'] = p_df.apply(classify_foc, axis=1)
    for cat in foc_cats:
        p_df[cat] = np.where(p_df['FOC類別'] == cat, p_df['實際FOC數量'], 0)
    
    p_df['FOC總量'] = p_df['實際FOC數量']
    p_df['收費量'] = np.where(p_df['金額'] > 0, p_df['數量'], 0)
    
    return p_df

# --- 3. 數據緩存讀取 ---
@st.cache_data(show_spinner="正在加載數據...")
def load_and_merge(file):
    all_sheets = pd.read_excel(file, sheet_name=None)
    data_list = [smart_normalize(df_temp) for name, df_temp in all_sheets.items() if not df_temp.empty]
    return pd.concat(data_list, ignore_index=True)

# --- 4. 側邊欄 ---
with st.sidebar:
    st.title("⚙️ 數據管理")
    admin_mode = st.toggle("管理員模式")
    uploaded_file = None
    if admin_mode:
        pwd = st.text_input("管理密碼", type="password")
        if pwd == "sunmax888":
            uploaded_file = st.file_uploader("上傳雙系統 Excel", type=["xlsx"])

# 決定來源
final_df = None
if uploaded_file:
    final_df = load_and_merge(uploaded_file)
elif os.path.exists("data.xlsx"):
    final_df = load_and_merge("data.xlsx")

st.title("📊 双美全球銷售數據")

if final_df is not None:
    df_all = final_df
    
    # 頂部篩選器
    sc1, sc2, sc3 = st.columns([1, 1, 2])
    sel_area = sc1.multiselect("🏙️ 區域", options=['台灣市場', '中國市場', '海外市場'], default=['海外市場', '中國市場'])
    available_countries = sorted(df_all[df_all['市場區域'].isin(sel_area)]['國家'].unique())
    sel_countries = sc3.multiselect("📍 國家", options=available_countries, default=available_countries)
    available_yrs = sorted([int(y) for y in df_all['年度'].unique() if y >= 2023])
    sel_yrs = sc2.multiselect("📅 年度", available_yrs, default=available_yrs)

    df = df_all[(df_all['年度'].isin(sel_yrs)) & (df_all['國家'].isin(sel_countries)) & (df_all['市場區域'].isin(sel_area))]

    if not df.empty:
        yr_str = ", ".join([f"{y}年" for y in sorted(sel_yrs)])
        st.write(f"🔍 **數據統計區間：{yr_str}**")

        # KPI
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("總銷售金額", f"NT${df['金額'].sum():,.0f}")
        k2.metric("收費訂單量", f"{df['收費量'].sum():,.0f}")
        k3.metric("FOC 贈送總量", f"{df['FOC總量'].sum():,.0f}")
        total_output = df['收費量'].sum() + df['FOC總量'].sum()
        k4.metric("平均贈針比", f"{(df['FOC總量'].sum() / (total_output + 0.0001) * 100):.1%}")
        k5.metric("活躍國家數", f"{df['國家'].nunique()}")

        tabs = st.tabs(["📊 市場佔比與YoY", "🎯 產品排行", "📉 FOC 專項分析", "⚡ 營運效率(進階)", "🔍 明細查詢"])

        # --- 分頁 4: 營運效率 (完整補回版) ---
        with tabs[3]:
            st.subheader("⚡ 全球營運效率與模式深度解析")
            
            c_eff1, c_eff2 = st.columns(2)
            
            with c_eff1:
                st.markdown("""<div class="report-note">
                <b>1. 市場滲透度分析 (活躍客戶數)：</b><br>
                ● <b>意義：</b> 日本(直營)數值越高代表開拓越多診所；經銷商模式通常為1，若變為0則需注意合約風險。<br>
                ● <b>分析重點：</b> 判斷該市場是靠「單一巨頭」還是「多點開花」。
                </div>""", unsafe_allow_html=True)
                active_cust = df.groupby(['國家', '商務模式'])['客戶'].nunique().reset_index()
                st.plotly_chart(px.bar(active_cust.sort_values('客戶', ascending=False), x='國家', y='客戶', color='商務模式', 
                                  title="各市場活躍客戶數 (滲透率)", text_auto=True), use_container_width=True)

            with c_eff2:
                st.markdown("""<div class="report-note">
                <b>2. 物流模式分析 (平均單筆採購量)：</b><br>
                ● <b>意義：</b> 經銷商應為「大宗進貨」(高數值)；日本診所為「小量多次」(低數值)。<br>
                ● <b>分析重點：</b> 若經銷商數值過低，代表物流報關次數多，營運成本被大幅侵蝕。
                </div>""", unsafe_allow_html=True)
                order_size = df[df['收費量']>0].groupby(['國家', '商務模式']).agg({'收費量':'sum', '銷貨日期':'count'}).reset_index()
                order_size['平均規模'] = (order_size['收費量'] / order_size['銷貨日期']).round(1)
                st.plotly_chart(px.bar(order_size.sort_values('平均規模', ascending=False), x='國家', y='平均規模', 
                                  color='商務模式', title="平均單次採購規模 (物流效率)", text_auto=True), use_container_width=True)
            
            st.divider()
            
            c_eff3, c_eff4 = st.columns([2, 1])
            with c_eff3:
                st.markdown("""<div class="report-note">
                <b>3. 行銷投資回報 (FOC 轉換效率)：</b><br>
                ● <b>意義：</b> 每投入1支贈針(FOC)，平均可以換回幾支收費訂單。<br>
                ● <b>分析重點：</b> 評估該國 Workshop 或 Sample 投入後的「變現力」。
                </div>""", unsafe_allow_html=True)
                eff_df = df.groupby('國家').agg({'收費量':'sum', 'FOC總量':'sum'}).reset_index()
                eff_df['效率'] = (eff_df['收費量'] / (eff_df['FOC總量'] + 0.001)).round(1)
                st.plotly_chart(px.bar(eff_df.sort_values('效率', ascending=False), x='國家', y='效率', 
                                  title="行銷槓桿比 (1支FOC換回幾單)", color='效率', text_auto=True), use_container_width=True)
            with c_eff4:
                st.markdown("""<div class="report-note">
                <b>4. 營收模式佔比：</b><br>
                顯示目前全球業務是依賴經銷商還是直營診所。
                </div>""", unsafe_allow_html=True)
                st.plotly_chart(px.pie(df, names='商務模式', hole=0.5, color_discrete_sequence=['#0984E3', '#00B894']), use_container_width=True)

            st.markdown("""<div class="report-note">
            <b>5. 全球採購季節性熱力圖：</b><br>
            ● <b>分析重點：</b> 顏色越深代表該月份進貨越多。用於追蹤展會後的補貨波段，協助海外部規劃年度出差。
            </div>""", unsafe_allow_html=True)
            heat_df = df.groupby(['國家', '月份'])['金額'].sum().reset_index()
            st.plotly_chart(px.density_heatmap(heat_df, x='月份', y='國家', z='金額', color_continuous_scale='Blues'), use_container_width=True)

        # --- 其餘分頁 (維持原本修正後的穩定功能) ---
        with tabs[0]: # 市場對比
            metric_opt = st.selectbox("選擇指標", ["銷售金額", "收費訂單數量", "FOC 總數量"])
            m_col = {'銷售金額': '金額', '收費訂單數量': '收費量', 'FOC 總數量': 'FOC總量'}[metric_opt]
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.pie(df.groupby('國家')[m_col].sum().reset_index(), values=m_col, names='國家', hole=0.4, title=f"各國份額", color_discrete_sequence=px.colors.qualitative.Pastel), use_container_width=True)
            pivot_df = df.pivot_table(index='國家', columns='年度', values=m_col, aggfunc='sum').fillna(0)
            sorted_yrs = sorted(pivot_df.columns)
            for i in range(1, len(sorted_yrs)):
                c, p = sorted_yrs[i], sorted_yrs[i-1]
                pivot_df[f"{c}年 成長%"] = np.where(pivot_df[p] == 0, np.nan, ((pivot_df[c] - pivot_df[p]) / pivot_df[p] * 100))
            
            def fmt_v(v, n): return f"{v:.1f}%" if '成長%' in str(n) and pd.notna(v) else ("-" if '成長%' in str(n) else f"{v:,.0f}")
            def sty_g(v, n): return f'color: {"red" if v>0 else "green"}; font-weight: bold' if '成長%' in str(n) and pd.notna(v) else ''
            
            # 整理年度與成長率排列
            final_cols = []
            for y in sorted_yrs:
                final_cols.append(y)
                if f"{y}年 成長%" in pivot_df.columns: final_cols.append(f"{y}年 成長%")
            
            disp_df = pivot_df[final_cols].copy()
            disp_df.columns = [f"{int(c)}年" if isinstance(c, (int, float)) else c for c in disp_df.columns]
            
            st.dataframe(disp_df.style.apply(lambda x: [sty_g(pivot_df.loc[x.name, col], col) for col in final_cols], axis=1).format(lambda v, c: fmt_v(v, c[1]), subset=disp_df.columns), use_container_width=True)

        with tabs[2]: # FOC 專項
            f_cols = ['培訓活動用針', '醫師酬勞', 'Workshop Training', 'Training/Rebate for JP', '市場贊助', '研究用針', '客訴補償', 'FOC樣品運輸']
            st.plotly_chart(px.bar(df[f_cols].sum().reset_index().rename(columns={'index':'類別', 0:'數量'}), x='類別', y='數量', color='類別', text_auto=True), use_container_width=True)
            target = st.selectbox("🔍 FOC 明細核對：", options=["全部 FOC"] + f_cols)
            foc_view = df[df['FOC總量']>0].copy()
            if target != "全部 FOC": foc_view = foc_view[foc_view['FOC類別'] == target]
            st.dataframe(foc_view[['銷貨日期', '國家', '客戶', '產品', '數量', '贈品量', '金額', '備註']], use_container_width=True)

        with tabs[4]: # 明細
            q = st.text_input("搜尋客戶或產品...").lower()
            q_df = df.copy()
            if q: q_df = q_df[q_df['客戶'].str.lower().str.contains(q, na=False) | q_df['產品'].str.lower().str.contains(q, na=False)]
            st.dataframe(q_df[['銷貨日期', '市場區域', '商務模式', '國家', '客戶', '產品', '數量', '金額', '備註']], use_container_width=True)
else:
    st.warning("👋 尚未發現數據檔案。請上傳 data.xlsx 或使用管理員模式導入。")
