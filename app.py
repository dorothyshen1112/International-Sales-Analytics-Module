import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# --- 1. 頁面配置 ---
st.set_page_config(page_title="雙美海外銷售分析系統", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF; color: #2D3436; }
    .stMetric { background-color: #F8F9FA; border-radius: 10px; padding: 15px; border: 1px solid #E9ECEF; }
    h1, h2, h3 { color: #0984E3; font-family: 'Helvetica Neue', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 數據處理 ---
def process_data(df):
    if df.shape[1] >= 76:
        raw_note = df.iloc[:, 75].astype(str).replace('nan', '')
    else:
        raw_note = df.iloc[:, -1].astype(str).replace('nan', '')

    df.columns = [str(c).replace(' ', '').replace('\n', '') for c in df.columns]
    df['主要備註'] = raw_note

    def find_col(targets):
        for t in targets:
            if t in df.columns: return t
        return df.columns[0]

    c_date = find_col(['銷貨日期', '單據日期', '銷貨日期A'])
    c_country = find_col(['國家', '地區'])
    c_customer = find_col(['客戶簡稱', '客戶', '送貨客戶全名'])
    c_product = find_col(['品名', '業務品名', '業務品號', '品號'])
    c_qty = find_col(['銷貨數量', '計價數量', '總數量'])
    c_price = find_col(['單價'])
    c_amount = find_col(['本幣未稅金額', '本幣合計'])

    df[c_date] = pd.to_datetime(df[c_date], errors='coerce')
    df['年度'] = df[c_date].dt.year
    df['月份'] = df[c_date].dt.month
    
    for c in [c_qty, c_price, c_amount]:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='coerce').fillna(0)

    # 產品名稱合併
    df[c_product] = df[c_product].replace('Sunmax DeusaDerm', 'Sunmax Deusaderm VITAL')

    # FOC 自動分類
    df['培訓與實操用針'] = 0
    df['醫師酬勞針'] = 0
    df['市場贊助與樣品'] = 0
    df['客訴補償'] = 0
    is_foc_item = (df[c_price] == 0) | (df[c_amount] == 0)
    
    kw_dict = {
        '培訓與實操用針': 'Workshop|實操|示範|培訓|講義|Demo',
        '醫師酬勞針': '酬勞|講師|醫師|陳咸伸|王柏鈞|技術費',
        '市場贊助與樣品': 'Sponsorship|Window|Sample|樣品|贊助|Influencer|廣告',
        '客訴補償': 'complaints|客訴|補償|瑕疵|更換|due to customer'
    }
    for key, kw in kw_dict.items():
        df.loc[is_foc_item & df['主要備註'].str.contains(kw, na=False, case=False), key] = df[c_qty]
    
    df['FOC總數量'] = df[['培訓與實操用針', '醫師酬勞針', '市場贊助與樣品', '客訴補償']].sum(axis=1)
    df['收費訂單量'] = np.where(df[c_amount] > 0, df[c_qty], 0)

    return df, {
        'date': c_date, 'country': c_country, 'customer': c_customer,
        'product': c_product, 'qty': c_qty, 'amount': c_amount, 'note': '主要備註'
    }

# --- 3. 側邊欄 ---
with st.sidebar:
    st.title("🛡️ 系統管理")
    admin_mode = st.toggle("管理員模式 (導入數據)")
    uploaded_file = None
    if admin_mode:
        pwd = st.text_input("管理密碼", type="password")
        if pwd == "sunmax888":
            uploaded_file = st.file_uploader("上傳 ERP Excel", type=["xlsx"])

st.title("🌐 海外銷售分析與預測系統")

if uploaded_file:
    raw_df = pd.read_excel(uploaded_file)
    df_all, m = process_data(raw_df)
    
    # 全域篩選器
    st.info("請設定篩選條件以連動所有分析：")
    sc1, sc2 = st.columns([1, 2])
    available_years = sorted([int(y) for y in df_all['年度'].dropna().unique()])
    selected_years = sc1.multiselect("📅 選擇年度", options=available_years, default=available_years)
    available_countries = sorted(df_all[m['country']].unique())
    selected_countries = sc2.multiselect("📍 選擇國家", options=available_countries, default=available_countries)

    df = df_all[(df_all['年度'].isin(selected_years)) & (df_all[m['country']].isin(selected_countries))]

    if not df.empty:
        # KPI 頂部摘要
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("累積營收", f"NT${df[m['amount']].sum():,.0f}")
        k2.metric("收費訂單量", f"{df['收費訂單量'].sum():,.0f}")
        k3.metric("FOC 總量", f"{df['FOC總數量'].sum():,.0f}")
        k4.metric("贈針比 (FOC %)", f"{(df['FOC總數量'].sum()/df[m['qty']].sum()*100):.1f}%")
        k5.metric("國家數", f"{df[m['country']].nunique()}")

        tabs = st.tabs(["🌍 市場成長對比", "🎯 產品分析", "📉 FOC 專項分析", "⚡ 營運效率(進階)", "🔍 數據明細"])

        with tabs[0]: # 市場對比
            compare_metric = st.selectbox("選擇分析指標", ["銷售金額 (台幣)", "收費訂單數量", "總出口件數", "FOC 總數量"])
            metric_map = {"銷售金額 (台幣)": m['amount'], "收費訂單數量": "收費訂單量", "總出口件數": m['qty'], "FOC 總數量": "FOC總數量"}
            target_col = metric_map[compare_metric]
            st.plotly_chart(px.bar(df.groupby(['年度', m['country']])[target_col].sum().reset_index(), x=m['country'], y=target_col, color='年度', barmode='group', text_auto='.2s'), use_container_width=True)
            
            # 多年度 YoY 表格
            pivot_df = df.pivot_table(index=m['country'], columns='年度', values=target_col, aggfunc='sum').fillna(0)
            display_df = pivot_df.copy()
            sorted_cols = sorted(pivot_df.columns)
            for i in range(1, len(sorted_cols)):
                curr, prev = sorted_cols[i], sorted_cols[i-1]
                display_df[f"{curr} 成長率(%)"] = ((pivot_df[curr] - pivot_df[prev]) / pivot_df[prev] * 100).replace([np.inf, -np.inf], 0).fillna(0)
            
            def style_growth(val, column_name):
                return f'color: {"green" if val > 0 else "red" if val < 0 else "black"}; font-weight: bold' if '成長率' in str(column_name) else ''
            st.dataframe(display_df.style.format("{:,.1f}").apply(lambda x: [style_growth(v, x.name) for v in x], axis=0))

        with tabs[1]: # 產品分析
            p_rank_col = m['amount'] if df[m['amount']].sum() > 0 else m['qty']
            p_df = df.groupby(m['product'])[p_rank_col].sum().sort_values(ascending=False).reset_index().head(12)
            st.plotly_chart(px.bar(p_df, x=p_rank_col, y=m['product'], orientation='h', color=p_rank_col, title="產品熱銷榜 (已合併名稱)"), use_container_width=True)

        with tabs[2]: # FOC 專項
            f_cols = ['培訓與實操用針', '醫師酬勞針', '市場贊助與樣品', '客訴補償']
            f_sum = df[f_cols].sum().reset_index()
            f_sum.columns = ['類別', '數量']
            st.plotly_chart(px.bar(f_sum, x='類別', y='數量', color='類別', text_auto=True), use_container_width=True)
            st.dataframe(df[df['FOC總數量']>0][[m['date'], m['country'], m['customer'], m['product'], m['qty'], '主要備註']], use_container_width=True)

        with tabs[3]: # 營運效率 (新設計)
            st.subheader("⚡ 營運效率進階指標")
            ec1, ec2 = st.columns(2)
            
            # 指標 1: FOC 轉換效率 (收費件數 / 贈針件數)
            eff_df = df.groupby(m['country']).agg({'收費訂單量':'sum', 'FOC總數量':'sum'}).reset_index()
            eff_df['轉換率(1支贈針換X支收費)'] = (eff_df['收費訂單量'] / eff_df['FOC總數量']).replace([np.inf, -np.inf], 0).fillna(0)
            fig_eff = px.bar(eff_df.sort_values('轉換率(1支贈針換X支收費)', ascending=False), 
                             x=m['country'], y='轉換率(1支贈針換X支收費)', 
                             title="FOC 轉換效率 (數值越高代表投入產出比越好)", color='轉換率(1支贈針換X支收費)')
            ec1.plotly_chart(fig_eff, use_container_width=True)
            
            # 指標 2: 物流效率 (平均每單採購量)
            logi_df = df[df['收費訂單量']>0].groupby(m['country']).agg({m['qty']:'sum', m['date']:'count'}).reset_index()
            logi_df['平均單筆採購量'] = logi_df[m['qty']] / logi_df[m['date']]
            fig_logi = px.bar(logi_df.sort_values('平均單筆採購量', ascending=False), 
                              x=m['country'], y='平均單筆採購量', 
                              title="物流效率 (平均每筆訂單採購件數)", color='平均單筆採購量', color_continuous_scale='Greens')
            ec2.plotly_chart(fig_logi, use_container_width=True)
            
            # 指標 3: 市場產品偏好 (LIDO 系列佔比)
            st.write("#### 產品策略：高階 (LIDO) 產品銷量佔比")
            df['Is_LIDO'] = df[m['product']].str.contains('LIDO', na=False)
            lido_df = df.groupby([m['country'], 'Is_LIDO'])['收費訂單量'].sum().unstack(fill_value=0).reset_index()
            if True in lido_df.columns:
                lido_df['LIDO佔比(%)'] = (lido_df[True] / (lido_df[True] + lido_df[False]) * 100).fillna(0)
                fig_lido = px.bar(lido_df.sort_values('LIDO佔比(%)', ascending=False), x=m['country'], y='LIDO佔比(%)', title="市場成熟度 (LIDO 產品銷量佔比)", color_discrete_sequence=['#FF7F0E'])
                st.plotly_chart(fig_lido, use_container_width=True)

        with tabs[4]: # 原始數據
            search_q = st.text_input("搜尋客戶或品名...").lower()
            q_df = df.copy()
            if search_q:
                q_df = q_df[q_df[m['customer']].str.lower().str.contains(search_q, na=False) | q_df[m['product']].str.lower().str.contains(search_q, na=False)]
            st.dataframe(q_df[[m['date'], m['country'], m['customer'], m['product'], m['qty'], m['amount'], '主要備註']], use_container_width=True)

else:
    st.info("💡 請管理員從左側側邊欄導入數據。")
