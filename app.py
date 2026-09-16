import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
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
    </style>
    """, unsafe_allow_html=True)

# --- 2. 智慧數據處理引擎 ---
def smart_normalize(df):
    if df.empty: return pd.DataFrame()
    bx_note_series = df.iloc[:, 75].astype(str).replace('nan', '') if df.shape[1] >= 76 else None
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
    p_df['月份'] = p_df['銷貨日期'].dt.month.fillna(0).astype(int)
    
    # 鎖定 2023+
    p_df = p_df[p_df['年度'] >= 2023].copy()
    if p_df.empty: return pd.DataFrame()

    p_df['國家'] = safe_get_series('COUNTRY').loc[p_df.index].fillna('台灣').astype(str)
    def classify_region(c):
        c_str = str(c).strip()
        if c_str in ['台灣', '臺灣', 'TAIWAN']: return '台灣市場'
        if c_str in ['大陸', '中國', 'CHINA', '中 國', '大 陸']: return '中國市場'
        return '海外市場'
    p_df['市場區域'] = p_df['國家'].apply(classify_region)
    
    p_df['客戶'] = safe_get_series('CUSTOMER').loc[p_df.index].fillna('未知客戶').astype(str)
    p_df['產品'] = safe_get_series('PRODUCT').loc[p_df.index].astype(str).str.replace('Sunmax DeusaDerm', 'Sunmax Deusaderm VITAL', case=False).str.replace('VITAL VITAL', 'VITAL', case=False)
    p_df['數量'] = safe_get_series('QTY', True).loc[p_df.index]
    p_df['贈品量_單獨欄位'] = safe_get_series('GIFT_QTY', True).loc[p_df.index]
    p_df['單價'] = safe_get_series('PRICE', True).loc[p_df.index]
    p_df['金額'] = safe_get_series('AMOUNT', True).loc[p_df.index]
    
    if bx_note_series is not None:
        p_df['備註'] = bx_note_series.loc[p_df.index].values
    else:
        p_df['備註'] = safe_get_series('NOTE').loc[p_df.index].astype(str).replace('nan', '')

    # FOC 邏輯
    p_df['實際FOC數量'] = np.where((p_df['單價'] <= 0) | (p_df['金額'] <= 0), p_df['數量'], 0) + p_df['贈品量_單獨欄位']
    
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
    foc_cats = ['培訓活動用針', '醫師酬勞', 'Workshop Training', 'Training/Rebate for JP', '市場贊助', '研究用針', '客訴補償', 'FOC樣品運輸']
    for cat in foc_cats:
        p_df[cat] = np.where(p_df['FOC類別'] == cat, p_df['實際FOC數量'], 0)
    
    p_df['FOC總量'] = p_df[foc_cats].sum(axis=1)
    p_df['收費量'] = np.where(p_df['金額'] > 0, p_df['數量'], 0)
    
    return p_df

# --- 3. UI 介面 ---
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
    
    sc1, sc2, sc3 = st.columns([1, 1, 2])
    sel_area = sc1.multiselect("🏙️ 區域", options=['台灣市場', '中國市場', '海外市場'], default=['海外市場', '中國市場'])
    filtered_by_area = df_all[df_all['市場區域'].isin(sel_area)]
    available_countries = sorted(filtered_by_area['國家'].unique())
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
        k3.metric("FOC 總量", f"{df['FOC總量'].sum():,.0f}")
        total_denom = df['收費量'].sum() + df['FOC總量'].sum()
        k4.metric("平均贈針比", f"{(df['FOC總量'].sum() / (total_denom + 0.0001) * 100):.1%}")
        k5.metric("活躍國家數", f"{df['國家'].nunique()}")

        tabs = st.tabs(["📊 市場佔比與YoY", "🎯 產品排行", "📉 FOC 專項分析", "⚡ 營運效率(進階)", "🔍 明細查詢"])

        # --- 營運效率分頁強化 ---
        with tabs[3]:
            st.subheader("⚡ 營運效率進階分析")
            
            c_eff1, c_eff2 = st.columns(2)
            
            with c_eff1:
                st.write("#### 1. 行銷槓桿效率 (每1支贈針帶動的營收)")
                # 計算每支贈針能換回多少台幣
                eff_data = df.groupby('國家').agg({'金額':'sum', 'FOC總量':'sum'}).reset_index()
                eff_data['槓桿率'] = (eff_data['金額'] / (eff_data['FOC總量'] + 0.001)).round(0)
                fig_lev = px.bar(eff_data.sort_values('槓桿率', ascending=False), x='國家', y='槓桿率', 
                                 title="行銷槓桿 (NTD / 1支FOC)", color='槓桿率', text_auto=True)
                st.plotly_chart(fig_lev, use_container_width=True)
                st.caption("※ 數值越高，代表該國贈針投入的產出價值越高。")

            with c_eff2:
                st.write("#### 2. 客戶集中度風險 (Top 1 客戶佔比)")
                # 計算每個國家第一大客戶的佔比
                def get_top1_share(group):
                    top1_val = group.groupby('客戶')['金額'].sum().max()
                    total_val = group['金額'].sum()
                    return (top1_val / (total_val + 0.001)) * 100
                
                risk_df = df[df['金額']>0].groupby('國家').apply(get_top1_share).reset_index()
                risk_df.columns = ['國家', 'Top1客戶佔比%']
                fig_risk = px.bar(risk_df.sort_values('Top1客戶佔比%', ascending=True), x='國家', y='Top1客戶佔比%', 
                                  title="客戶集中度 (越高代表越依賴單一代理商)", color='Top1客戶佔比%', color_continuous_scale='Reds')
                st.plotly_chart(fig_risk, use_container_width=True)

            st.write("#### 3. 全球採購季節性熱力圖 (進貨節奏分析)")
            heat_df = df.groupby(['國家', '月份'])['金額'].sum().reset_index()
            # 補齊 1-12 月
            all_months = pd.DataFrame({'月份': range(1, 13)})
            fig_heat = px.density_heatmap(heat_df, x='月份', y='國家', z='金額', 
                                          color_continuous_scale='Viridis', title="各國進貨熱點 (月份 vs 國家)")
            st.plotly_chart(fig_heat, use_container_width=True)

        # 其餘分頁 (維持原本優秀功能)
        with tabs[0]: # YoY 成長
            metric_opt = st.selectbox("選擇指標", ["銷售金額", "收費訂單數量", "FOC 總數量"])
            m_col = {'銷售金額': '金額', '收費訂單數量': '收費量', 'FOC 總數量': 'FOC總量'}[metric_opt]
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.pie(df.groupby('國家')[m_col].sum().reset_index(), values=m_col, names='國家', hole=0.4, title=f"各國份額", color_discrete_sequence=px.colors.qualitative.Pastel), use_container_width=True)
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
            def format_v(val, col_n):
                if '成長率%' in str(col_n): return f"{val:.1f}%" if pd.notna(val) else "-"
                return f"{val:,.0f}"
            def style_g(val, col_n):
                if '成長率%' in str(col_n):
                    if pd.isna(val): return 'color: gray'
                    return f'color: {"red" if val > 0 else "green"}; font-weight: bold'
                return ''
            formatted_df = display_pivot.copy().astype(object)
            for col in display_pivot.columns: formatted_df[col] = display_pivot[col].apply(lambda x: format_v(x, col))
            st.dataframe(formatted_df.style.apply(lambda x: [style_g(display_pivot.loc[x.name, col], col) for col in display_pivot.columns], axis=1), use_container_width=True)

        with tabs[1]:
            p_df = df.groupby('產品')['金額'].sum().sort_values(ascending=False).reset_index().head(12)
            st.plotly_chart(px.bar(p_df, x='金額', y='產品', orientation='h', color='金額', title="全球產品銷售排行榜"), use_container_width=True)

        with tabs[2]:
            f_cols = ['培訓活動用針', '醫師酬勞', 'Workshop Training', 'Training/Rebate for JP', '市場贊助', '研究用針', '客訴補償', 'FOC樣品運輸']
            f_sum = df[f_cols].sum().reset_index()
            f_sum.columns = ['類別', '數量']
            st.plotly_chart(px.bar(f_sum, x='類別', y='數量', color='類別', text_auto=True), use_container_width=True)
            target_foc_cat = st.selectbox("🔍 FOC 類別明細核對：", options=["全部 FOC"] + f_cols)
            foc_display_df = df[df['FOC總量'] > 0].copy()
            if target_foc_cat != "全部 FOC": foc_display_df = foc_display_df[foc_display_df['FOC類別'] == target_foc_cat]
            st.dataframe(foc_display_df[['銷貨日期', '國家', '客戶', '產品', '數量', '贈品量_單獨欄位', '金額', '備註']], use_container_width=True)

        with tabs[4]:
            search_q = st.text_input("搜尋客戶、產品或備註...").lower()
            q_df = df.copy()
            if search_q: q_df = q_df[q_df['客戶'].str.lower().str.contains(search_q, na=False) | q_df['產品'].str.lower().str.contains(search_q, na=False) | q_df['備註'].str.lower().str.contains(search_q, na=False)]
            st.dataframe(q_df[['銷貨日期', '市場區域', '國家', '客戶', '產品', '數量', '金額', '備註']], use_container_width=True)
else:
    st.info("👋 管理員您好，請導入 Excel。本版已強化營運效率分析指標。")
