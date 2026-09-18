import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
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
    .report-note { 
        background-color: #E3F2FD; padding: 15px; border-left: 6px solid #2196F3; 
        border-radius: 4px; margin-bottom: 15px; font-size: 0.95rem; color: #1565C0; line-height: 1.6;
    }
    .highlight-text { color: #D32F2F; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 智慧數據處理引擎 ---
def smart_normalize(df):
    if df is None or df.empty or len(df.columns) < 2: return pd.DataFrame()
    
    # 預抓新系統 BX 備註 (索引 75)
    bx_note = df.iloc[:, 75].astype(str).replace('nan', '') if df.shape[1] >= 76 else None
    
    # 清理標題
    df.columns = [str(c).strip().replace('\n', '').upper() for c in df.columns]
    
    # [地毯式匹配字典]
    m = {
        'DATE': ['銷貨日期', '單據日期', '日期', 'DATE', '銷貨日期A', '成交日期', '開單日期'],
        'YEAR': ['年度', '年', 'YEAR', '年份', '西元年', 'YYYY'],
        'COUNTRY': ['國家', '地區', 'COUNTRY', '區域', '廠別名稱', '地  區', '省份', '市場', '收貨地址', '送貨地址一'],
        'CUSTOMER': ['客戶簡稱', '客戶', '客戶名稱', 'CUSTOMER', '送貨客戶全名', '經銷商'],
        'PRODUCT': ['品名', '業務品名', '產品名稱', 'PRODUCT', '品號', '項目名稱'],
        'QTY': ['銷貨數量', '數量', '計價數量', '總數量', 'QTY', '件數', '出貨數量'],
        'GIFT_QTY': ['贈/備品量', '贈品量', '贈品數量', 'GIFTQTY'],
        'PRICE': ['單價', '單價NT', 'PRICE', '單   價'],
        'AMOUNT': ['本幣未稅金額', '本幣合計', '金額', '未稅金額', 'AMOUNT', '總計', '本幣金額', '銷貨金額', '合計']
    }

    def find_col(key):
        for name in m.get(key, []):
            for real_col in df.columns:
                if name == real_col or name in real_col: return real_col
        return None

    p_df = pd.DataFrame()
    
    # 1. 時間識別
    c_year = find_col('YEAR')
    c_date = find_col('DATE')
    if c_year:
        p_df['年度'] = pd.to_numeric(df[c_year].astype(str).str.extract('(\d{4})')[0], errors='coerce').fillna(0).astype(int)
        p_df['銷貨日期'] = pd.to_datetime(p_df['年度'].astype(str) + '-01-01')
    elif c_date:
        dates = pd.to_datetime(df[c_date], errors='coerce')
        p_df['銷貨日期'] = dates
        p_df['年度'] = dates.dt.year.fillna(0).astype(int)
    else: return pd.DataFrame()

    p_df = p_df[(p_df['年度'] >= 2000) & (p_df['年度'] < 2040)].copy()
    if p_df.empty: return pd.DataFrame()
    p_df['月份'] = p_df['銷貨日期'].dt.month.fillna(1).astype(int)

    # 2. 數值提取 (強力清理)
    def to_num(key):
        col = find_col(key)
        if not col: return pd.Series([0.0]*len(df)).loc[p_df.index]
        return pd.to_numeric(df[col].loc[p_df.index].astype(str).str.replace(r'[^-0-9.]', '', regex=True), errors='coerce').fillna(0.0)

    p_df['數量'] = to_num('QTY')
    p_df['贈品量'] = to_num('GIFT_QTY')
    p_df['單價'] = to_num('PRICE')
    p_df['金額'] = to_num('AMOUNT')

    # 3. 智慧區域分類 (根據國家、客戶、或地址自動判定)
    c_country_col = find_col('COUNTRY')
    c_cust_col = find_col('CUSTOMER')
    
    raw_country = df[c_country_col].astype(str).loc[p_df.index] if c_country_col else pd.Series(['']*len(p_df))
    raw_cust = df[c_cust_col].astype(str).loc[p_df.index] if c_cust_col else pd.Series(['']*len(p_df))

    def auto_classify(row_idx):
        country_txt = raw_country.loc[row_idx].upper()
        cust_txt = raw_cust.loc[row_idx].upper()
        full_txt = country_txt + cust_txt
        
        if any(x in full_txt for x in ['台灣', '臺灣', 'TAIWAN']): return '台灣', '台灣市場', '經銷商模式'
        if any(x in full_txt for x in ['大陸', '中國', 'CHINA', 'MAINLAND']): return '大陸', '中國市場', '經銷商模式'
        if 'SINGAPORE' in full_txt or '新加坡' in full_txt: return '新加坡', '海外市場', '經銷商模式'
        if 'MALAYSIA' in full_txt or '馬來西亞' in full_txt: return '馬來西亞', '海外市場', '經銷商模式'
        if 'PHILIPPINES' in full_txt or '菲律賓' in full_txt: return '菲律賓', '海外市場', '經銷商模式'
        if 'THAILAND' in full_txt or '泰國' in full_txt: return '泰國', '海外市場', '經銷商模式'
        if 'JAPAN' in full_txt or '日本' in full_txt: return '日本', '海外市場', '直營診所模式'
        if 'GERMANY' in full_txt or '德國' in full_txt: return '德國', '海外市場', '經銷商模式'
        return '其他', '海外市場', '經銷商模式'

    p_df['國家'], p_df['市場區域'], p_df['商務模式'] = zip(*[auto_classify(i) for i in p_df.index])

    # 4. 客戶全名對接
    def get_official(name, country):
        n, c = str(name).upper(), str(country).upper()
        if 'VANGUARD' in n:
            if 'SINGAPORE' in c or '新加坡' in c: return 'VANGUARD AESTHETICS PTE. LTD.'
            if 'PHILIPPINES' in c or '菲律賓' in c: return 'VANGUARD AESTHETICS OPC'
            if 'MALAYSIA' in c or '馬來西亞' in c: return 'Vanguard Aesthetics Sdn Bhd'
        if 'QUALTECH' in n: return 'Qualtech Consulting (Thailand)'
        return re.sub(r'\s*(PTE\.?\s*LTD\.?|SDN\.?\s*BHD\.?|OPC|CORP\.?|INC\.?|CO\.?|LTD\.?)$', '', n).strip()
    
    p_df['客戶'] = [get_official(n, c) for n, c in zip(raw_cust, p_df['國家'])]
    p_df['產品'] = df[find_col('PRODUCT')].loc[p_df.index].astype(str).str.replace('Sunmax DeusaDerm', 'Sunmax Deusaderm VITAL', case=False).str.replace('VITAL VITAL', 'VITAL', case=False) if find_col('PRODUCT') else "未知產品"
    p_df['備註'] = bx_note.loc[p_df.index].values if bx_note is not None else (df[find_col('NOTE')].loc[p_df.index].astype(str).replace('nan', '') if find_col('NOTE') else "")

    # 5. FOC 邏輯修正 (金額大於0 絕對不是 FOC)
    p_df['實際FOC數量'] = np.where((p_df['金額'] <= 0) & ((p_df['單價'] <= 0) | (p_df['數量'] > 0)), p_df['數量'], 0.0) + p_df['贈品量']
    # 如果舊系統沒抓到單價但金額很高，強制將 FOC 設為 0
    p_df.loc[p_df['金額'] > 0, '實際FOC數量'] = 0.0
    
    foc_cats = ['培訓活動用針', '醫師酬勞', 'Workshop Training', 'Training/Rebate for JP', '市場贊助', '研究用針', '客訴補償', 'FOC樣品運輸']
    for cat in foc_cats: p_df[cat] = 0.0
    def classify_foc_row(row):
        f_qty = row['實際FOC數量']
        if f_qty <= 0: return '非FOC'
        rem = str(row['備註']).lower()
        if 'workshop training' in rem: return 'Workshop Training'
        if any(x in rem for x in ['實操針', '示範針', '會員實操針', '培訓針', '培訓施打用針', 'demo', '培訓活動施打用針']): return '培訓活動用針'
        if any(x in rem for x in ['酬勞針', '講師針', 'speech']): return '醫師酬勞'
        if any(x in rem for x in ['rebate', 'training']): return 'Training/Rebate for JP'
        if any(x in rem for x in ['sponsorship', 'launch event', 'launch']): return '市場贊助'
        if any(x in rem for x in ['sample needles', 'irb']): return '研究用針'
        if any(x in rem for x in ['complaints', '客訴', '補償', '瑕疵', '更換', 'quality', 'customer complaints']): return '客訴補償'
        return 'FOC樣品運輸'
    p_df['FOC類別'] = p_df.apply(classify_foc_row, axis=1)
    for cat in foc_cats: p_df[cat] = np.where(p_df['FOC類別'] == cat, p_df['實際FOC數量'], 0.0)
    
    p_df['FOC總量'] = p_df['實際FOC數量']
    p_df['收費量'] = np.where(p_df['金額'] > 0, p_df['數量'], 0.0)
    return p_df

# --- 3. 數據加載 ---
def load_all_data():
    if not os.path.exists("data.xlsx"): return None
    try:
        sheets = pd.read_excel("data.xlsx", sheet_name=None)
        all_dfs = [smart_normalize(df_sheet) for name, df_sheet in sheets.items() if not df_sheet.empty]
        return pd.concat(all_dfs, ignore_index=True) if all_dfs else None
    except Exception as e:
        st.error(f"❌ 數據加載失敗：{e}")
        return None

# --- 4. 主介面 ---
with st.sidebar:
    st.title("⚙️ 數據管理")
    if st.toggle("管理員模式") and st.text_input("密碼", type="password") == "sunmax888":
        st.file_uploader("上傳 Excel", type=["xlsx"])
        if st.button("更新數據快取"): st.cache_data.clear()

st.title("📊 双美全球銷售數據")
df_all = load_all_data()

if df_all is not None:
    sc1, sc2, sc3 = st.columns([1, 1, 2])
    sel_area = sc1.multiselect("區域", ['台灣市場', '中國市場', '海外市場'], default=['海外市場', '中國市場', '台灣市場'])
    available_countries = sorted(df_all[df_all['市場區域'].isin(sel_area)]['國家'].unique())
    sel_countries = sc3.multiselect("國家", available_countries, default=available_countries)
    all_yrs = sorted([int(y) for y in df_all['年度'].unique() if y > 0])
    sel_yrs = sc2.multiselect("年度", options=all_yrs, default=all_yrs)

    df = df_all[(df_all['年度'].isin(sel_yrs)) & (df_all['國家'].isin(sel_countries)) & (df_all['市場區域'].isin(sel_area))]

    if not df.empty:
        yr_label = f"{min(sel_yrs)}年-{max(sel_yrs)}年" if len(sel_yrs)>1 else f"{sel_yrs[0]}年"
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("總銷售金額", f"NT${df['金額'].sum():,.0f}")
        k2.metric("收費訂單量", f"{df['收費量'].sum():,.0f}")
        k3.metric("FOC 贈送總量", f"{df['FOC總量'].sum():,.0f}")
        # --- 關鍵修正：修正贈針比計算邏輯 ---
        tot_qty = df['收費量'].sum() + df['FOC總量'].sum()
        f_rate = (df['FOC總量'].sum() / (tot_qty + 0.0001)) * 100
        k4.metric("平均贈針比", f"{f_rate:.1f}%")
        k5.metric("活躍國家數", f"{df['國家'].nunique()}")

        tabs = st.tabs(["📊 市場佔比與YoY", "🎯 產品排行", "📉 FOC 專項分析", "⚡ 營運效率(進階)", "🔍 明細查詢"])
        
        with tabs[0]: # 市場佔比 & 大表格
            metric_opt = st.selectbox("分析指標", ["金額", "收費量", "FOC總量"])
            m_col = {'金額': '金額', '收費量': '收費量', 'FOC總量': 'FOC總量'}[metric_opt]
            st.plotly_chart(px.pie(df.groupby('國家')[m_col].sum().reset_index(), values=m_col, names='國家', hole=0.4, title="各國份額佔比", color_discrete_sequence=px.colors.qualitative.Pastel), use_container_width=True)
            st.divider()
            pivot = df.pivot_table(index='國家', columns='年度', values=m_col, aggfunc='sum').fillna(0)
            sorted_y = sorted(pivot.columns); disp = pivot.copy()
            for i in range(1, len(sorted_y)):
                curr, prev = sorted_y[i], sorted_y[i-1]
                disp[f"{curr}年 成長%"] = np.where(pivot[prev]==0, np.nan, (pivot[curr]-pivot[prev])/pivot[prev]*100)
            f_cols = []
            f_dict = {}
            for y in sorted_y:
                y_c = f"{int(y)}年"; disp = disp.rename(columns={y: y_c})
                f_cols.append(y_c); f_dict[y_c] = "{:,.0f}"
                g_c = f"{int(y)}年 成長%"
                if g_c in disp.columns: f_cols.append(g_c); f_dict[g_c] = "{:.1f}%"
            def color_logic(v, n):
                if '成長%' in str(n) and pd.notna(v):
                    return 'color: #D32F2F; font-weight: bold' if v > 0.001 else 'color: #388E3C; font-weight: bold'
                return ''
            st.dataframe(disp[f_cols].style.format(f_dict, na_rep="-").apply(lambda x: [color_logic(v, x.name) for v in x], axis=0), height=450, use_container_width=True)

        with tabs[3]: # 營運效率 (藍色說明永久鎖定)
            st.subheader("⚡ 全球營運效率與模式深度解析")
            st.markdown('<div class="report-note"><b>1. 市場滲透度分析：</b><br>● 日本(直營)顯示診所總量。經銷商國家應為 1。</div>', unsafe_allow_html=True)
            st.plotly_chart(px.bar(df.groupby(['國家', '商務模式'])['客戶'].nunique().reset_index().sort_values('客戶', ascending=False), x='國家', y='客戶', color='商務模式', title="活躍客戶/診所總數", text_auto=True), use_container_width=True)
            with st.expander("📋 查看名單"):
                cust_dt = df.groupby(['國家', '商務模式'])['客戶'].unique().reset_index()
                cust_dt['清單'] = cust_dt['客戶'].apply(lambda x: "\n".join([f"• {name}" for name in x]))
                st.write(cust_dt[['國家', '商務模式', '清單']].to_html(escape=False).replace('\\n', '<br>'), unsafe_allow_html=True)
            st.divider()
            st.markdown('<div class="report-note"><b>2. 物流模式分析：</b><br>● 經銷商大宗進貨數值高；日本診所小量多次數值低。</div>', unsafe_allow_html=True)
            order_sz = df[df['收費量']>0].groupby(['國家', '商務模式']).agg({'收費量':'sum', '銷貨日期':'count'}).reset_index()
            order_sz['規模'] = (order_sz['收費量'] / (order_sz['銷貨日期'] + 0.0001)).round(1)
            st.plotly_chart(px.bar(order_sz.sort_values('規模', ascending=False), x='國家', y='規模', color='商務模式', title="平均單次訂單規模", text_auto=True), use_container_width=True)
            st.divider()
            st.markdown('<div class="report-note"><b>3. 行銷投資回報：</b><br>● 每投入1支贈針換回幾張訂單。數值越高代表投資報酬率越高。</div>', unsafe_allow_html=True)
            eff_df = df.groupby('國家').agg({'收費量':'sum', 'FOC總量':'sum'}).reset_index()
            eff_df['效率'] = (eff_df['收費量'] / (eff_df['FOC總量'] + 0.001)).round(1)
            st.plotly_chart(px.bar(eff_df.sort_values('效率', ascending=False), x='國家', y='效率', title="行銷槓桿比", color='效率', text_auto=True), use_container_width=True)
            st.divider()
            st.markdown('<div class="report-note"><b>4. 銷售季節性分析：</b><br>顏色越深代表進貨越多。</div>', unsafe_allow_html=True)
            heat_df = df.groupby(['國家', '月份'])['金額'].sum().reset_index()
            fig_h = px.density_heatmap(heat_df, x='月份', y='國家', z='金額', color_continuous_scale='Blues', nbinsx=12, range_x=[0.5, 12.5], text_auto='.2s')
            fig_h.update_xaxes(tickmode='linear', tick0=1, dtick=1, ticktext=[f"{i}月" for i in range(1,13)], tickvals=list(range(1,13)))
            st.plotly_chart(fig_h, use_container_width=True)

        with tabs[2]: # FOC
            f_cols = ['培訓活動用針', '醫師酬勞', 'Workshop Training', 'Training/Rebate for JP', '市場贊助', '研究用針', '客訴補償', 'FOC樣品運輸']
            st.plotly_chart(px.bar(df[f_cols].sum().reset_index().rename(columns={'index':'類別', 0:'數量'}), x='類別', y='數量', color='類別', text_auto=True), use_container_width=True)
            target = st.selectbox("🔍 FOC 明細過濾：", options=["全部 FOC"] + f_cols)
            foc_v = df[df['FOC總量']>0].copy()
            if target != "全部 FOC": foc_v = foc_v[foc_v['FOC類別'] == target]
            st.dataframe(foc_v[['銷貨日期', '國家', '客戶', '產品', '數量', '贈品量', '金額', '備註']], use_container_width=True)
        with tabs[4]: # 明細
            q = st.text_input("搜尋關鍵字...").lower()
            q_df = df.copy()
            if q: q_df = q_df[q_df['客戶'].str.lower().str.contains(q, na=False) | q_df['產品'].str.lower().str.contains(q, na=False)]
            st.dataframe(q_df[['銷貨日期', '市場區域', '商務模式', '國家', '客戶', '產品', '數量', '金額', '備註']], use_container_width=True)
else:
    st.warning("👋 尚未發現數據檔案。請上傳 data.xlsx。")
