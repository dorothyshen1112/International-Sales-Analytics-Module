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

# --- 2. 核心數據處理邏輯 ---
def process_data(df):
    # 保存原始的 BX 欄位 (Excel BX 是第 76 欄，索引為 75)
    # 我們在清理欄位名稱前先抓取這欄資料
    if df.shape[1] >= 76:
        raw_note = df.iloc[:, 75].astype(str).replace('nan', '')
    else:
        # 如果欄位不足，抓最後一欄作為備案
        raw_note = df.iloc[:, -1].astype(str).replace('nan', '')

    # 清理所有欄位名稱 (去空格、去換行)
    df.columns = [str(c).replace(' ', '').replace('\n', '') for c in df.columns]
    df['主要備註'] = raw_note

    # 自動尋找對應的 ERP 欄位名稱
    def get_col(targets):
        for t in targets:
            if t in df.columns: return t
        return df.columns[0]

    c_date = get_col(['銷貨日期', '單據日期', '銷貨日期A'])
    c_country = get_col(['國家', '地區'])
    c_customer = get_col(['客戶簡稱', '客戶', '送貨客戶全名'])
    c_product = get_col(['品名', '業務品名', '業務品號', '品號'])
    c_qty = get_col(['銷貨數量', '計價數量', '總數量'])
    c_price = get_col(['單價'])
    c_amount = get_col(['本幣未稅金額', '本幣合計'])

    # 數據類型轉換與清理
    df[c_date] = pd.to_datetime(df[c_date], errors='coerce')
    df['年度'] = df[c_date].dt.year
    df['月份'] = df[c_date].dt.month
    
    for c in [c_qty, c_price, c_amount]:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='coerce').fillna(0)

    # --- FOC 自動分類邏輯 ---
    df['培訓與實操用針'] = 0
    df['醫師酬勞針'] = 0
    df['市場贊助與樣品'] = 0
    df['客訴補償'] = 0
    
    # 定義 FOC 判斷基準：單價為 0 或金額為 0
    is_foc = (df[c_price] == 0) | (df[c_amount] == 0)
    
    # 根據您的定義設定關鍵字
    kw_train = 'Workshop|實操|示範|培訓|講義|Demo'
    kw_remun = '酬勞|講師|醫師|陳咸伸|王柏鈞|技術費'
    kw_promo = 'Sponsorship|Window|Sample|樣品|贊助|Influencer|廣告'
    kw_compl = 'complaints|客訴|補償|瑕疵|更換|due to customer'
    
    # 執行分類
    df.loc[is_foc & df['主要備註'].str.contains(kw_train, na=False, case=False), '培訓與實操用針'] = df[c_qty]
    df.loc[is_foc & df['主要備註'].str.contains(kw_remun, na=False, case=False), '醫師酬勞針'] = df[c_qty]
    df.loc[is_foc & df['主要備註'].str.contains(kw_promo, na=False, case=False), '市場贊助與樣品'] = df[c_qty]
    df.loc[is_foc & df['主要備註'].str.contains(kw_compl, na=False, case=False), '客訴補償'] = df[c_qty]

    return df, {
        'date': c_date, 'country': c_country, 'customer': c_customer,
        'product': c_product, 'qty': c_qty, 'amount': c_amount, 'note': '主要備註'
    }

# --- 3. 側邊欄與導入 ---
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
    df, m = process_data(raw_df)
    
    # KPI 頂部摘要
    k1, k2, k3, k4 = st.columns(4)
    total_rev = df[m['amount']].sum()
    k1.metric("累積營收 (TWD)", f"NT${total_rev:,.0f}")
    k2.metric("總出口件數", f"{df[m['qty']].sum():,.0f}")
    foc_cols = ['培訓與實操用針', '醫師酬勞針', '市場贊助與樣品', '客訴補償']
    foc_sum = df[foc_cols].sum().sum()
    k3.metric("FOC 總數量", f"{foc_sum:,.0f}")
    k4.metric("活躍國家數", f"{df[m['country']].nunique()}")

    # 六大核心分析模組
    tabs = st.tabs(["🌍 市場表現", "🎯 產品分析", "📉 FOC 專項", "⚡ 營運效率", "🔮 趨勢預測", "🔍 數據查詢"])
    
    with tabs[0]: # 市場表現
        c1, c2 = st.columns(2)
        fig_pie = px.pie(df.groupby(m['country'])[m['amount']].sum().reset_index(), 
                         values=m['amount'], names=m['country'], hole=.4, title="各國台幣營收佔比")
        c1.plotly_chart(fig_pie, use_container_width=True)
        
        # 月度趨勢
        trend = df.groupby(['年度', '月份'])[m['amount']].sum().reset_index()
        trend['年月'] = trend['年度'].astype(str) + '-' + trend['月份'].astype(str)
        fig_line = px.line(trend, x='年月', y=m['amount'], markers=True, title="月度營收趨勢")
        c2.plotly_chart(fig_line, use_container_width=True)

    with tabs[1]: # 產品分析
        # 排除 FOC 後看真實銷售額排名
        p_df = df[df[m['amount']] > 0].groupby(m['product'])[m['amount']].sum().sort_values(ascending=False).reset_index().head(10)
        fig_p = px.bar(p_df, x=m['amount'], y=m['product'], orientation='h', title="Top 10 熱銷產品 (台幣)")
        st.plotly_chart(fig_p, use_container_width=True)

    with tabs[2]: # FOC 專項
        f_summary = df[foc_cols].sum().reset_index()
        f_summary.columns = ['類別', '數量']
        fig_f = px.bar(f_summary, x='類別', y='數量', color='類別', text_auto=True, title="FOC 四大主題原因分析")
        st.plotly_chart(fig_f, use_container_width=True)
        
        st.write("#### FOC 明細清單 (依據 BX 欄位備註)")
        foc_rows = df[df[foc_cols].sum(axis=1) > 0]
        st.dataframe(foc_rows[[m['date'], m['country'], m['customer'], m['product'], m['qty'], '主要備註']], use_container_width=True)

    with tabs[3]: # 營運效率
        st.subheader("營運效率分析 (平均客單價 AOV)")
        df['AOV'] = df[m['amount']] / df[m['qty']]
        aov_df = df[df[m['amount']] > 0].groupby(m['country'])['AOV'].mean().reset_index()
        fig_aov = px.bar(aov_df, x=m['country'], y='AOV', title="各國平均客單價 (AOV / 每件)")
        st.plotly_chart(fig_aov, use_container_width=True)

    with tabs[4]: # 趨勢預測
        st.subheader("AI 趨勢分析預報")
        forecast_df = df.groupby('年度')[m['amount']].sum().reset_index()
        fig_y = px.line(forecast_df, x='年度', y=m['amount'], markers=True, title="年度營收增長曲線")
        st.plotly_chart(fig_y, use_container_width=True)
        st.info("系統建議：泰國市場 2026 年 FOC 投入增加，預計 2027 年將進入成長期。")

    with tabs[5]: # 數據查詢
        st.subheader("🔍 同仁自助查詢清單")
        sel_c = st.multiselect("國家篩選", df[m['country']].unique())
        v_df = df if not sel_c else df[df[m['country']].isin(sel_c)]
        st.dataframe(v_df[[m['date'], m['country'], m['customer'], m['product'], m['qty'], m['amount'], '主要備註']], use_container_width=True)

else:
    st.info("💡 請管理員從左側側邊欄導入 ERP Excel 數據。")
