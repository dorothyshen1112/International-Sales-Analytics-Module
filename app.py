import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
import re

# --- 1. 頁面配置 ---
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

# --- 2. 智慧數據引擎 ---
def smart_normalize(df):
    if df.empty: return pd.DataFrame()
    bx_note_series = df.iloc[:, 75].astype(str).replace('nan', '') if df.shape[1] >= 76 else None
    df.columns = [str(c).replace(' ', '').replace('\n', '').upper() for c in df.columns]
    mapping = {
        'DATE': ['銷貨日期', '單據日期', '日期', 'DATE', '銷貨日期A'],
        'COUNTRY': ['國家', '地區', 'COUNTRY', '區域', '廠別名稱'],
        'CUSTOMER': ['客戶簡稱', '客戶', '客戶名稱', 'CUSTOMER', '送貨客戶全名'],
        'PRODUCT': ['品名', '業務品名', '產品名稱', 'PRODUCT', '品號'],
        'QTY': ['銷貨數量', '數量', '計價數量', '總數量', 'QTY'],
        'GIFT_QTY': ['贈/備品量', '贈品量', '贈品數量', 'GIFTQTY'],
        'PRICE': ['單價', '單價NT', 'PRICE'],
        'AMOUNT': ['本幣未稅金額', '本幣合計', '金額', '未稅金額', 'AMOUNT']
    }
    def find_best(ts):
        for t in ts:
            if t.upper() in df.columns: return t.upper()
        return None
    def safe_get(key, is_num=False):
        col = find_best(mapping[key])
        if col:
            if is_num: return pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0.0)
            return df[col]
        return pd.Series([0.0 if is_num else ""] * len(df))

    p_df = pd.DataFrame()
    c_date = find_best(mapping['DATE'])
    p_df['銷貨日期'] = pd.to_datetime(df[c_date], errors='coerce') if c_date else pd.Timestamp.now()
    p_df['年度'] = p_df['銷貨日期'].dt.year.fillna(0).astype(int)
    p_df['月份'] = p_df['銷貨日期'].dt.month.fillna(0).astype(int)
    p_df = p_df[p_df['年度'] >= 2023].copy()
    if p_df.empty: return pd.DataFrame()

    p_df['國家'] = safe_get('COUNTRY').loc[p_df.index].fillna('台灣').astype(str)
    def classify_region_and_model(c):
        c_str = str(c).strip()
        if any(x in c_str for x in ['台灣', '臺灣', 'TAIWAN']): return '台灣市場', '經銷商模式'
        if any(x in c_str for x in ['大陸', '中國', 'CHINA']): return '中國市場', '經銷商模式'
        if any(x in c_str for x in ['日本', 'JAPAN']): return '海外市場', '直營診所模式'
        return '海外市場', '經銷商模式'
    p_df['市場區域'], p_df['商務模式'] = zip(*p_df['國家'].apply(classify_region_and_model))
    
    # 官方全名對接
    def get_official(name, country):
        n_up, c_up = str(name).upper(), str(country).upper()
        if 'VANGUARD' in n_up:
            if '新加坡' in country or 'SINGAPORE' in c_up: return 'VANGUARD AESTHETICS PTE. LTD.'
            if '菲律賓' in country or 'PHILIPPINES' in c_up: return 'VANGUARD AESTHETICS OPC'
            if '馬來西亞' in country or 'MALAYSIA' in c_up: return 'Vanguard Aesthetics Sdn Bhd'
        if 'QUALTECH' in n_up: return 'Qualtech Consulting (Thailand)'
        return re.sub(r'\s*(PTE\.?\s*LTD\.?|SDN\.?\s*BHD\.?|OPC|CORP\.?|INC\.?|CO\.?|LTD\.?)$', '', n_up).strip()
    p_df['客戶'] = [get_official(n, c) for n, c in zip(safe_get('CUSTOMER').loc[p_df.index], p_df['國家'])]
    
    p_df['產品'] = safe_get('PRODUCT').loc[p_df.index].astype(str).str.replace('Sunmax DeusaDerm', 'Sunmax Deusaderm VITAL', case=False).str.replace('VITAL VITAL', 'VITAL', case=False)
    p_df['數量'] = safe_get('QTY', True).loc[p_df.index]
    p_df['贈品量'] = safe_get('GIFT_QTY', True).loc[p_df.index]
    p_df['單價'] = safe_get('PRICE', True).loc[p_df.index]
    p_df['金額'] = safe_get('AMOUNT', True).loc[p_df.index]
    p_df['備註'] = bx_note_series.loc[p_df.index].values if bx_note_series is not None else safe_get('NOTE').loc[p_df.index].astype(str).replace('nan', '')

    p_df['實際FOC數量'] = np.where((p_df['單價'] <= 0) | (p_df['金額'] <= 0), p_df['數量'], 0.0) + p_df['贈品量']
    foc_cats = ['培訓活動用針', '醫師酬勞', 'Workshop Training', 'Training/Rebate for JP', '市場贊助', '研究用針', '客訴補償', 'FOC樣品運輸']
    for cat in foc_cats: p_df[cat] = 0.0
    def classify_foc(row):
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
    p_df['FOC類別'] = p_df.apply(classify_foc, axis=1)
    for cat in foc_cats: p_df[cat] = np.where(p_df['FOC類別'] == cat, p_df['實際FOC數量'], 0.0)
    p_df['FOC總量'] = p_df['實際FOC數量']
    p_df['收費量'] = np.where(p_df['金額'] > 0, p_df['數量'], 0.0)
    return p_df

# --- 3. 數據加載 (含診斷邏輯) ---
def load_and_merge(file):
    try:
        all_sheets = pd.read_excel(file, sheet_name=None)
        data_list = [smart_normalize(df_temp) for name, df_temp in all_sheets.items() if not df_temp.empty]
        return pd.concat(data_list, ignore_index=True) if data_list else None
    except Exception as e:
        st.error(f"讀取檔案出錯: {e}")
        return None

# --- 4. 側邊欄 ---
with st.sidebar:
    st.title("⚙️ 數據管理")
    admin_mode = st.toggle("管理員模式")
    uploaded_file = None
    if admin_mode:
        pwd = st.text_input("管理密碼", type="password")
        if pwd == "sunmax888":
            uploaded_file = st.file_uploader("上傳最新 Excel (臨時)", type=["xlsx"])
            if st.button("清除快取並重整"):
                st.cache_data.clear()
                st.rerun()

# 核心邏輯：自動尋找 data.xlsx
final_df = None
if uploaded_file:
    final_df = load_and_merge(uploaded_file)
elif os.path.exists("data.xlsx"):
    final_df = load_and_merge("data.xlsx")
else:
    # 診斷模式：列出目前目錄下的所有檔案，幫管理員抓 Bug
    files_in_dir = os.listdir(".")
    st.warning(f"⚠️ 找不到 `data.xlsx`。目前資料夾內的檔案有：{files_in_dir}")
    st.info("💡 請確保您上傳到 GitHub 的檔案名稱精確為 `data.xlsx` (全小寫)。")

st.title("📊 双美全球銷售數據")

if final_df is not None:
    df_all = final_df
    sc1, sc2, sc3 = st.columns([1, 1, 2])
    sel_area = sc1.multiselect("區域", options=['台灣市場', '中國市場', '海外市場'], default=['海外市場', '中國市場'])
    available_countries = sorted(df_all[df_all['市場區域'].isin(sel_area)]['國家'].unique())
    sel_countries = sc3.multiselect("國家", available_countries, default=available_countries)
    available_yrs = sorted([int(y) for y in df_all['年度'].unique() if y >= 2023])
    sel_yrs = sc2.multiselect("年度", available_yrs, default=available_yrs)
    df = df_all[(df_all['年度'].isin(sel_yrs)) & (df_all['國家'].isin(sel_countries)) & (df_all['市場區域'].isin(sel_area))]

    if not df.empty:
        yr_str = ", ".join([f"{y}年" for y in sorted(sel_yrs)])
        st.write(f"🔍 **數據統計區間：{yr_str}**")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("總銷售金額", f"NT${df['金額'].sum():,.0f}")
        k2.metric("收費訂單量", f"{df['收費量'].sum():,.0f}")
        k3.metric("FOC 贈送總量", f"{df['FOC總量'].sum():,.0f}")
        total_out = df['收費量'].sum() + df['FOC總量'].sum()
        k4.metric("平均贈針比", f"{(df['FOC總量'].sum() / (total_out + 0.0001) * 100):.1%}")
        k5.metric("活躍國家數", f"{df['國家'].nunique()}")

        tabs = st.tabs(["📊 市場佔比與YoY", "🎯 產品排行", "📉 FOC 專項分析", "⚡ 營運效率(進階)", "🔍 明細查詢"])
        
        with tabs[0]: # YoY
            metric_opt = st.selectbox("分析指標", ["銷售金額", "收費訂單數量", "FOC 總數量"])
            m_col = {'銷售金額': '金額', '收費訂單數量': '收費量', 'FOC 總數量': 'FOC總量'}[metric_opt]
            c1, c2 = st.columns([1, 1.2])
            with c1: st.plotly_chart(px.pie(df.groupby('國家')[m_col].sum().reset_index(), values=m_col, names='國家', hole=0.4, title=f"各國份額佔比", color_discrete_sequence=px.colors.qualitative.Pastel), use_container_width=True)
            with c2:
                pivot_df = df.pivot_table(index='國家', columns='年度', values=m_col, aggfunc='sum').fillna(0)
                sorted_years = sorted(pivot_df.columns); display_data = pivot_df.copy()
                for i in range(1, len(sorted_years)):
                    curr, prev = sorted_years[i], sorted_years[i-1]
                    display_data[f"{curr}年 成長%"] = np.where(pivot_df[prev] == 0, np.nan, (pivot_df[curr] - pivot_df[prev]) / pivot_df[prev] * 100)
                final_cols = []
                for y in sorted_years:
                    y_col = f"{int(y)}年"; display_data = display_data.rename(columns={y: y_col})
                    final_cols.append(y_col)
                    if f"{int(y)}年 成長%" in display_data.columns: final_cols.append(f"{int(y)}年 成長%")
                def color_rg(val):
                    if pd.isna(val): return 'color: #9E9E9E'
                    if isinstance(val, (int, float)):
                        if val > 0.001: return 'color: #D32F2F; font-weight: bold'
                        if val < -0.001: return 'color: #388E3C; font-weight: bold'
                    return ''
                st.write("#### 逐年成長趨勢表 (YoY%)")
                st_df = display_data[final_cols].style.format(lambda v: f"{v:.1f}%" if (isinstance(v, float) and not v.is_integer() and v < 1000) else f"{v:,.0f}", na_rep="-")
                try: st_df = st_df.map(color_rg)
                except: st_df = st_df.applymap(color_rg)
                st.dataframe(st_df, use_container_width=True)

        with tabs[1]: st.plotly_chart(px.bar(df.groupby('產品')['金額'].sum().sort_values(ascending=False).reset_index().head(12), x='金額', y='產品', orientation='h', color='金額', title="全球產品銷售排行榜"), use_container_width=True)
        
        with tabs[2]: # FOC
            f_cols = ['培訓活動用針', '醫師酬勞', 'Workshop Training', 'Training/Rebate for JP', '市場贊助', '研究用針', '客訴補償', 'FOC樣品運輸']
            st.plotly_chart(px.bar(df[f_cols].sum().reset_index().rename(columns={'index':'類別', 0:'數量'}), x='類別', y='數量', color='類別', text_auto=True), use_container_width=True)
            target = st.selectbox("🔍 FOC 明細過濾：", options=["全部 FOC"] + f_cols)
            foc_view = df[df['FOC總量']>0].copy()
            if target != "全部 FOC": foc_view = foc_view[foc_view['FOC類別'] == target]
            st.dataframe(foc_view[['銷貨日期', '國家', '客戶', '產品', '數量', '贈品量', '金額', '備註']], use_container_width=True)

        with tabs[3]: # 營運效率
            st.subheader("⚡ 全球營運效率與模式深度解析")
            c_e1, c_e2 = st.columns(2)
            with c_e1:
                st.markdown('<div class="report-note"><b>1. 市場滲透度分析 (活躍客戶數)：</b><br>● 顯示已進入的經銷商或直營診所總數。</div>', unsafe_allow_html=True)
                active_count = df.groupby(['國家', '商務模式'])['客戶'].nunique().reset_index()
                st.plotly_chart(px.bar(active_count.sort_values('客戶', ascending=False), x='國家', y='客戶', color='商務模式', title="活躍客戶/診所總數", text_auto=True), use_container_width=True)
                with st.expander("📋 點此展開：查看各國具體客戶/診所完整名單"):
                    cust_detail = df.groupby(['國家', '商務模式'])['客戶'].unique().reset_index()
                    cust_detail['官方全名/診所名單'] = cust_detail['客戶'].apply(lambda x: "\n".join([f"• {name}" for name in x]))
                    st.write(cust_detail[['國家', '商務模式', '官方全名/診所名單']].to_html(escape=False).replace('\\n', '<br>'), unsafe_allow_html=True)
            with c_e2:
                st.markdown('<div class="report-note"><b>2. 物流模式分析 (平均單次規模)：</b><br>● 經銷商大宗採購 vs 日本診所小量多次。</div>', unsafe_allow_html=True)
                order_size = df[df['收費量']>0].groupby(['國家', '商務模式']).agg({'收費量':'sum', '銷貨日期':'count'}).reset_index()
                order_size['規模'] = (order_size['收費量'] / (order_size['銷貨日期'] + 0.0001)).round(1)
                st.plotly_chart(px.bar(order_size.sort_values('規模', ascending=False), x='國家', y='規模', color='商務模式', title="平均單次訂單規模", text_auto=True), use_container_width=True)
            
            st.divider()
            c_e3, c_e4 = st.columns([2, 1])
            with c_e3:
                st.markdown('<div class="report-note"><b>3. 行銷投資回報 (FOC 轉換效率)：</b><br>● 每投入1支贈針換回幾張訂單。數值越高代表投資報酬率越高。</div>', unsafe_allow_html=True)
                eff_df = df.groupby('國家').agg({'收費量':'sum', 'FOC總量':'sum'}).reset_index()
                eff_df['效率'] = (eff_df['收費量'] / (eff_df['FOC總量'] + 0.001)).round(1)
                st.plotly_chart(px.bar(eff_df.sort_values('效率', ascending=False), x='國家', y='效率', title="行銷槓桿比 (1支FOC換回幾張訂單)", color='效率', text_auto=True), use_container_width=True)
            with c_e4:
                st.plotly_chart(px.pie(df, names='商務模式', hole=0.5, color_discrete_sequence=['#0984E3', '#00B894'], title="全球模式佔比"), use_container_width=True)
            
            heat_df = df.groupby(['國家', '月份'])['金額'].sum().reset_index()
            fig_heat = px.density_heatmap(heat_df, x='月份', y='國家', z='金額', color_continuous_scale='Blues', nbinsx=12, range_x=[0.5, 12.5], text_auto='.2s')
            fig_heat.update_xaxes(tickmode='linear', tick0=1, dtick=1, ticktext=[f"{i}月" for i in range(1,13)], tickvals=list(range(1,13)))
            st.plotly_chart(fig_heat, use_container_width=True)

        with tabs[4]: # 明細
            q = st.text_input("搜尋關鍵字...").lower()
            q_df = df.copy()
            if q: q_df = q_df[q_df['客戶'].str.lower().str.contains(q, na=False) | q_df['產品'].str.lower().str.contains(q, na=False)]
            st.dataframe(q_df[['銷貨日期', '市場區域', '商務模式', '國家', '客戶', '產品', '數量', '金額', '備註']], use_container_width=True)
else:
    st.stop()
