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

    # 數據清洗
    df[c_date] = pd.to_datetime(df[c_date], errors='coerce')
    df['年度'] = df[c_date].dt.year
    df['月份'] = df[c_date].dt.month
    
    for c in [c_qty, c_price, c_amount]:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='coerce').fillna(0)

    # --- 需求 3：產品名稱合併 ---
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
    
    # --- 需求 2：全域置頂篩選器 ---
    st.info("請設定下方篩選條件以連動所有分析分頁：")
    sc1, sc2 = st.columns([1, 2])
    available_years = sorted([int(y) for y in df_all['年度'].dropna().unique()])
    selected_years = sc1.multiselect("📅 選擇分析年度", options=available_years, default=available_years[-2:])
    available_countries = sorted(df_all[m['country']].unique())
    selected_countries = sc2.multiselect("📍 選擇分析國家", options=available_countries, default=available_countries)

    # 應用篩選
    df = df_all[(df_all['年度'].isin(selected_years)) & (df_all[m['country']].isin(selected_countries))]

    if df.empty:
        st.warning("⚠️ 此篩選範圍下無數據。")
    else:
        # KPI 頂部摘要
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("累積營收", f"NT${df[m['amount']].sum():,.0f}")
        k2.metric("總出口件數", f"{df[m['qty']].sum():,.0f}")
        k3.metric("收費訂單量", f"{df[df[m['amount']] > 0][m['qty']].sum():,.0f}")
        k4.metric("FOC 總量", f"{df['FOC總數量'].sum():,.0f}")
        k5.metric("國家數", f"{df[m['country']].nunique()}")

        # 分頁系統
        tabs = st.tabs(["🌍 市場成長對比", "🎯 產品分析", "📉 FOC 專項", "⚡ 營運效率", "🔍 原始數據"])

        with tabs[0]: # 市場對比
            st.subheader("📈 跨年度指標對比與 YoY %")
            # 需求 1：指標加入 FOC
            compare_metric = st.selectbox("選擇分析指標", ["銷售金額 (台幣)", "收費訂單數量", "總出口件數", "FOC 總數量"])
            metric_map = {"銷售金額 (台幣)": m['amount'], "收費訂單數量": m['qty'], "總出口件數": m['qty'], "FOC 總數量": "FOC總數量"}
            target_col = metric_map[compare_metric]

            chart_data = df.groupby(['年度', m['country']])[target_col].sum().reset_index()
            fig = px.bar(chart_data, x=m['country'], y=target_col, color='年度', barmode='group', text_auto='.2s', title=f"{compare_metric} 年度比較")
            st.plotly_chart(fig, use_container_width=True)

            # 計算 YoY
            st.write("#### 數據摘要與成長分析")
            pivot_df = df.pivot_table(index=m['country'], columns='年度', values=target_col, aggfunc='sum').fillna(0)
            
            if len(selected_years) >= 2:
                latest, prev = max(selected_years), sorted(selected_years)[-2]
                pivot_df['成長率(%)'] = ((pivot_df[latest] - pivot_df[prev]) / pivot_df[prev] * 100).replace([np.inf, -np.inf], 0).fillna(0)
                
                # 安全的染色函數 (相容新版 Pandas)
                def color_logic(val):
                    if isinstance(val, (int, float)):
                        return 'color: green' if val > 0 else 'color: red' if val < 0 else 'color: black'
                    return ''
                
                st.dataframe(pivot_df.style.format("{:,.1f}").apply(lambda x: [color_logic(v) if x.name == '成長率(%)' else '' for v in x]))
            else:
                st.dataframe(pivot_df.style.format("{:,.0f}"))

        with tabs[1]: # 產品分析 (自動連動年份)
            st.subheader(f"🎯 產品銷售排名 ({selected_years}年)")
            rank_col = m['amount'] if df[m['amount']].sum() > 0 else m['qty']
            p_df = df.groupby(m['product'])[rank_col].sum().sort_values(ascending=False).reset_index().head(12)
            fig_p = px.bar(p_df, x=rank_col, y=m['product'], orientation='h', color=rank_col, title="Top 12 產品熱銷榜")
            st.plotly_chart(fig_p, use_container_width=True)

        with tabs[2]: # FOC 專項 (自動連動年份)
            st.subheader(f"📉 FOC 原因分佈 ({selected_years}年)")
            f_cols = ['培訓與實操用針', '醫師酬勞針', '市場贊助與樣品', '客訴補償']
            f_sum = df[f_cols].sum().reset_index()
            f_sum.columns = ['類別', '數量']
            st.plotly_chart(px.bar(f_sum, x='類別', y='數量', color='類別', text_auto=True), use_container_width=True)
            st.dataframe(df[df['FOC總數量']>0][[m['date'], m['country'], m['customer'], m['product'], m['qty'], '主要備註']], use_container_width=True)

        with tabs[3]: # 營運效率 (自動連動年份)
            st.subheader("⚡ 營運效率 (AOV)")
            paid_df = df[df[m['amount']] > 0].copy()
            if not paid_df.empty:
                paid_df['AOV'] = paid_df[m['amount']] / paid_df[m['qty']]
                st.plotly_chart(px.bar(paid_df.groupby(m['country'])['AOV'].mean().reset_index(), x=m['country'], y='AOV', color='AOV', title="各國平均客單價"), use_container_width=True)
            else:
                st.warning("無付費訂單數據。")

        with tabs[4]: # 原始數據
            search_q = st.text_input("搜尋客戶或品名...").lower()
            q_df = df.copy()
            if search_q:
                q_df = q_df[q_df[m['customer']].str.lower().str.contains(search_q, na=False) | q_df[m['product']].str.lower().str.contains(search_q, na=False)]
            st.dataframe(q_df[[m['date'], m['country'], m['customer'], m['product'], m['qty'], m['amount'], '主要備註']], use_container_width=True)

else:
    st.info("💡 請管理員從左側側邊欄導入數據。")
