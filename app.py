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

# --- 2. 核心數據處理 ---
def process_data(df):
    # BX 欄位抓取 (第 76 欄)
    if df.shape[1] >= 76:
        raw_note = df.iloc[:, 75].astype(str).replace('nan', '')
    else:
        raw_note = df.iloc[:, -1].astype(str).replace('nan', '')

    # 清理欄位空格
    df.columns = [str(c).replace(' ', '').replace('\n', '') for c in df.columns]
    df['主要備註'] = raw_note

    # 自動匹配欄位
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

    # 數據清洗與轉換
    df[c_date] = pd.to_datetime(df[c_date], errors='coerce')
    df['年度'] = df[c_date].dt.year
    df['月份'] = df[c_date].dt.month
    
    for c in [c_qty, c_price, c_amount]:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='coerce').fillna(0)

    # 產品名稱合併邏輯 (VITAL)
    df[c_product] = df[c_product].replace('Sunmax DeusaDerm', 'Sunmax Deusaderm VITAL')
    
    # --- 新增：市場區域劃分 ---
    df['市場區域'] = df[c_country].apply(lambda x: '台灣市場' if str(x) == '台灣' else '海外市場')

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

# --- 3. 側邊欄與管理 ---
with st.sidebar:
    st.title("🛡️ 系統管理")
    admin_mode = st.toggle("管理員模式 (導入數據)")
    uploaded_file = None
    if admin_mode:
        pwd = st.text_input("管理密碼", type="password")
        if pwd == "sunmax888":
            uploaded_file = st.file_uploader("上傳 ERP Excel (支援包含台灣之數據)", type=["xlsx"])

st.title("🌐 雙美全球銷售分析與預測系統")

if uploaded_file:
    raw_df = pd.read_excel(uploaded_file)
    df_all, m = process_data(raw_df)
    
    # --- 全域置頂篩選器 ---
    st.info("💡 設定篩選條件，連動下方所有分析圖表：")
    sc1, sc2, sc3 = st.columns([1, 1, 2])
    
    # 市場區域篩選 (台灣/海外)
    selected_areas = sc1.multiselect("🏙️ 市場區域", options=['台灣市場', '海外市場'], default=['海外市場'])
    
    # 年度篩選
    available_years = sorted([int(y) for y in df_all['年度'].dropna().unique()])
    selected_years = sc2.multiselect("📅 分析年度", options=available_years, default=available_years)
    
    # 國家篩選 (根據區域自動更新選項)
    temp_df = df_all[df_all['市場區域'].isin(selected_areas)]
    available_countries = sorted(temp_df[m['country']].unique())
    selected_countries = sc3.multiselect("📍 分析國家", options=available_countries, default=available_countries)

    # 最終應用篩選
    df = df_all[
        (df_all['年度'].isin(selected_years)) & 
        (df_all[m['country']].isin(selected_countries)) & 
        (df_all['市場區域'].isin(selected_areas))
    ]

    if not df.empty:
        # KPI 頂部摘要
        k1, k2, k3, k4, k5 = st.columns(5)
        total_revenue = df[m['amount']].sum()
        k1.metric("所選區域營收", f"NT${total_revenue:,.0f}")
        k2.metric("收費訂單量", f"{df['收費訂單量'].sum():,.0f}")
        k3.metric("FOC 總數量", f"{df['FOC總數量'].sum():,.0f}")
        k4.metric("平均贈針比", f"{(df['FOC總數量'].sum()/(df[m['qty']].sum()+0.0001)*100):.1f}%")
        k5.metric("國家/區域數", f"{df[m['country']].nunique()}")

        tabs = st.tabs(["📊 市場佔比與成長", "🎯 產品排名", "📉 FOC 專項", "⚡ 營運效率", "🔍 原始數據"])

        with tabs[0]: # 市場佔比與成長
            st.subheader("🌍 全球市場權重與成長分析")
            c1, c2 = st.columns([1, 1])
            
            # 指標選擇
            compare_metric = st.selectbox("選擇分析指標", ["銷售金額 (台幣)", "收費訂單數量", "FOC 總數量"])
            metric_map = {"銷售金額 (台幣)": m['amount'], "收費訂單數量": "收費訂單量", "FOC 總數量": "FOC總數量"}
            target_col = metric_map[compare_metric]

            # 左側：各國佔比圓餅圖 (Pie Chart)
            pie_df = df.groupby(m['country'])[target_col].sum().reset_index()
            fig_pie = px.pie(pie_df, values=target_col, names=m['country'], hole=.4, 
                             title=f"各國家/地區 {compare_metric} 佔比 (%)",
                             color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            c1.plotly_chart(fig_pie, use_container_width=True)
            
            # 右側：年度成長對比長條圖
            chart_data = df.groupby(['年度', m['country']])[target_col].sum().reset_index()
            fig_bar = px.bar(chart_data, x=m['country'], y=target_col, color='年度', barmode='group', 
                             text_auto='.2s', title="年度指標成長對比")
            c2.plotly_chart(fig_bar, use_container_width=True)

            # 下方：逐年成長趨勢表 (YoY)
            st.write("#### 逐年成長趨勢摘要 (YoY %)")
            pivot_df = df.pivot_table(index=m['country'], columns='年度', values=target_col, aggfunc='sum').fillna(0)
            display_df = pivot_df.copy()
            sorted_cols = sorted(pivot_df.columns)
            for i in range(1, len(sorted_cols)):
                curr, prev = sorted_cols[i], sorted_cols[i-1]
                display_df[f"{curr} 成長率(%)"] = ((pivot_df[curr] - pivot_df[prev]) / (pivot_df[prev]+0.0001) * 100).replace([np.inf, -np.inf], 0).fillna(0)
            
            # 重新排列欄位 (年度1, 年度2, 成長%, 年度3, 成長%)
            ordered_cols = []
            for yr in sorted_cols:
                ordered_cols.append(yr)
                g_col = f"{yr} 成長率(%)"
                if g_col in display_df.columns: ordered_cols.append(g_col)
            
            def style_growth(val, column_name):
                if '成長率' in str(column_name):
                    color = 'green' if val > 0 else 'red' if val < 0 else 'black'
                    return f'color: {color}; font-weight: bold'
                return ''

            st.dataframe(display_df[ordered_cols].style.format("{:,.1f}").apply(lambda x: [style_growth(v, x.name) for v in x], axis=0))

        with tabs[1]: # 產品分析
            p_rank_col = m['amount'] if df[m['amount']].sum() > 0 else m['qty']
            p_df = df.groupby(m['product'])[p_rank_col].sum().sort_values(ascending=False).reset_index().head(12)
            st.plotly_chart(px.bar(p_df, x=p_rank_col, y=m['product'], orientation='h', color=p_rank_col, title="Top 12 產品熱銷榜 (全球)"), use_container_width=True)

        with tabs[2]: # FOC 專項
            f_cols = ['培訓與實操用針', '醫師酬勞針', '市場贊助與樣品', '客訴補償']
            f_sum = df[f_cols].sum().reset_index()
            f_sum.columns = ['類別', '數量']
            st.plotly_chart(px.bar(f_sum, x='類別', y='數量', color='類別', text_auto=True, title="全球 FOC 配置原因"), use_container_width=True)
            st.dataframe(df[df['FOC總數量']>0][[m['date'], m['country'], m['customer'], m['product'], m['qty'], '主要備註']], use_container_width=True)

        with tabs[3]: # 營運效率
            st.subheader("⚡ 營運效率進階指標")
            ec1, ec2 = st.columns(2)
            # 指標 1: FOC 轉換效率
            eff_df = df.groupby(m['country']).agg({'收費訂單量':'sum', 'FOC總數量':'sum'}).reset_index()
            eff_df['轉換率'] = (eff_df['收費訂單量'] / (eff_df['FOC總數量']+0.0001)).fillna(0)
            ec1.plotly_chart(px.bar(eff_df.sort_values('轉換率', ascending=False), x=m['country'], y='轉換率', title="FOC 帶動銷量效率 (每支贈針換回幾支訂單)", color='轉換率'), use_container_width=True)
            # 指標 2: 區域營收貢獻度
            area_df = df.groupby('市場區域')[m['amount']].sum().reset_index()
            ec2.plotly_chart(px.pie(area_df, values=m['amount'], names='市場區域', title="國內 vs 海外 營收佔比", hole=0.5), use_container_width=True)

        with tabs[4]: # 原始數據
            search_q = st.text_input("搜尋客戶或品名...").lower()
            q_df = df.copy()
            if search_q:
                q_df = q_df[q_df[m['customer']].str.lower().str.contains(search_q, na=False) | q_df[m['product']].str.lower().str.contains(search_q, na=False)]
            st.dataframe(q_df[[m['date'], m['country'], m['customer'], m['product'], m['qty'], m['amount'], '主要備註']], use_container_width=True)

else:
    st.info("💡 請管理員從左側側邊欄導入數據（支援包含「台灣」在內的 ERP 全球明細）。")
