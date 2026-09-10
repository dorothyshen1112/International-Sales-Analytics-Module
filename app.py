import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# --- 1. 頁面配置 (精密淨白風格) ---
st.set_page_config(page_title="雙美海外銷售分析系統", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF; color: #2D3436; }
    .stMetric { background-color: #F8F9FA; border-radius: 10px; padding: 15px; border: 1px solid #E9ECEF; }
    [data-testid="stMetricValue"] { font-size: 1.8rem !important; }
    h1, h2, h3 { color: #0984E3; font-family: 'Helvetica Neue', sans-serif; }
    div.stTabs [data-baseweb="tab-list"] { gap: 15px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 核心數據處理 ---
def process_data(df):
    # 抓取 BX 欄位 (第 76 欄，索引 75)
    if df.shape[1] >= 76:
        raw_note = df.iloc[:, 75].astype(str).replace('nan', '')
    else:
        raw_note = df.iloc[:, -1].astype(str).replace('nan', '')

    # 清理欄位名稱 (ERP 常見空格問題)
    df.columns = [str(c).replace(' ', '').replace('\n', '') for c in df.columns]
    df['主要備註'] = raw_note

    # 自動匹配關鍵欄位
    def find_c(targets):
        for t in targets:
            if t in df.columns: return t
        return df.columns[0]

    c_date = find_c(['銷貨日期', '單據日期', '銷貨日期A'])
    c_country = find_c(['國家', '地區'])
    c_customer = find_c(['客戶簡稱', '客戶', '送貨客戶全名'])
    c_product = find_c(['品名', '業務品名', '業務品號', '品號', '業務品名_1'])
    c_qty = find_c(['銷貨數量', '計價數量', '總數量'])
    c_price = find_c(['單價'])
    c_amount = find_c(['本幣未稅金額', '本幣合計'])

    # 數據轉型與清理
    df[c_date] = pd.to_datetime(df[c_date], errors='coerce')
    df['年度'] = df[c_date].dt.year
    df['月份'] = df[c_date].dt.month
    
    for c in [c_qty, c_price, c_amount]:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='coerce').fillna(0)

    # FOC 四大類別定義
    df['培訓與實操用針'] = 0
    df['醫師酬勞針'] = 0
    df['市場贊助與樣品'] = 0
    df['客訴補償'] = 0
    
    # 判定標準：單價=0 或 金額=0
    is_foc_item = (df[c_price] == 0) | (df[c_amount] == 0)
    
    kw_dict = {
        '培訓與實操用針': 'Workshop|實操|示範|培訓|講義|Demo',
        '醫師酬勞針': '酬勞|講師|醫師|陳咸伸|王柏鈞|技術費',
        '市場贊助與樣品': 'Sponsorship|Window|Sample|樣品|贊助|Influencer|廣告',
        '客訴補償': 'complaints|客訴|補償|瑕疵|更換|due to customer'
    }
    
    for key, kw in kw_dict.items():
        df.loc[is_foc_item & df['主要備註'].str.contains(kw, na=False, case=False), key] = df[c_qty]

    # 計算一個總 FOC 欄位方便對比
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
    df, m = process_data(raw_df)
    
    # --- 指標卡片 ---
    k1, k2, k3, k4, k5 = st.columns(5)
    total_rev = df[m['amount']].sum()
    k1.metric("累積營收 (TWD)", f"NT${total_rev:,.0f}")
    
    total_export = df[m['qty']].sum()
    k2.metric("總出口件數", f"{total_export:,.0f}")
    
    paid_qty = df[df[m['amount']] > 0][m['qty']].sum()
    k3.metric("訂單數量 (收費)", f"{paid_qty:,.0f}")
    
    foc_sum = df['FOC總數量'].sum()
    k4.metric("FOC 總數量", f"{foc_sum:,.0f}")
    k5.metric("活躍國家數", f"{df[m['country']].nunique()}")

    # --- 分頁分析 ---
    tabs = st.tabs(["🌍 市場對比分析", "🎯 產品分析", "📉 FOC 專項", "⚡ 營運效率", "🔍 數據查詢"])
    
    with tabs[0]: # 市場對比
        st.subheader("📈 跨維度與市場綜合對比")
        c1, c2, c3 = st.columns([1, 1, 2])
        compare_metric = c1.selectbox("選擇分析指標", ["銷售金額 (台幣)", "收費訂單數量", "總出口件數", "FOC 總數量"])
        
        metric_map = {
            "銷售金額 (台幣)": m['amount'],
            "收費訂單數量": m['qty'],
            "總出口件數": m['qty'],
            "FOC 總數量": "FOC總數量"
        }
        target_col = metric_map[compare_metric]
        
        selected_years = c2.multiselect("年度", options=sorted(df['年度'].dropna().unique()), default=df['年度'].dropna().unique()[-2:])
        selected_countries = c3.multiselect("國家", options=df[m['country']].unique(), default=df[m['country']].unique()[:3])
        
        comp_df = df[(df['年度'].isin(selected_years)) & (df[m['country']].isin(selected_countries))]
        
        if not comp_df.empty:
            fig_compare = px.bar(comp_df.groupby(['年度', m['country']])[target_col].sum().reset_index(), 
                                 x=m['country'], y=target_col, color='年度', barmode='group',
                                 text_auto='.2s', title=f"{compare_metric} 對比分析")
            st.plotly_chart(fig_compare, use_container_width=True)
            
            # 數據表
            pivot_df = comp_df.pivot_table(index=m['country'], columns='年度', values=target_col, aggfunc='sum').fillna(0)
            st.write("#### 數據明細表")
            st.table(pivot_df.style.format("{:,.0f}"))

    with tabs[1]: # 產品分析
        st.subheader("🎯 產品銷售與分佈排行")
        # 增加判斷：如果金額全為 0 (只有FOC)，則改按數量排
        rank_col = m['amount'] if df[m['amount']].sum() > 0 else m['qty']
        p_df = df.groupby(m['product'])[rank_col].sum().sort_values(ascending=False).reset_index().head(10)
        
        if not p_df.empty:
            fig_p = px.bar(p_df, x=rank_col, y=m['product'], orientation='h', color=rank_col, 
                           title=f"Top 10 產品排行 (按{'金額' if rank_col==m['amount'] else '數量'})")
            st.plotly_chart(fig_p, use_container_width=True)
        else:
            st.warning("目前篩選條件下無產品數據")

    with tabs[2]: # FOC 專項
        st.subheader("📉 FOC 資源分配細目")
        foc_cols = ['培訓與實操用針', '醫師酬勞針', '市場贊助與樣品', '客訴補償']
        f_summary = df[foc_cols].sum().reset_index()
        f_summary.columns = ['類別', '數量']
        fig_f = px.bar(f_summary, x='類別', y='數量', color='類別', text_auto=True)
        st.plotly_chart(fig_f, use_container_width=True)
        
        st.write("#### FOC 識別明細 (來源：BX 欄位)")
        st.dataframe(df[df['FOC總數量'] > 0][[m['date'], m['country'], m['customer'], m['product'], m['qty'], '主要備註']], use_container_width=True)

    with tabs[3]: # 營運效率
        st.subheader("⚡ 營運效率指標 (AOV)")
        # 只計算收費訂單的 AOV
        paid_df = df[df[m['amount']] > 0].copy()
        if not paid_df.empty:
            paid_df['AOV'] = paid_df[m['amount']] / paid_df[m['qty']]
            aov_df = paid_df.groupby(m['country'])['AOV'].mean().reset_index()
            fig_aov = px.bar(aov_df, x=m['country'], y='AOV', title="各國平均每件收費針劑之台幣價值", color='AOV')
            st.plotly_chart(fig_aov, use_container_width=True)
        else:
            st.warning("數據中無收費訂單，無法計算營運效率 (AOV)")

    with tabs[4]: # 數據查詢
        st.subheader("🔍 同仁自助查詢")
        search_q = st.text_input("輸入關鍵字搜尋 (客戶簡稱或品名)...").lower()
        q_df = df.copy()
        if search_q:
            q_df = q_df[q_df[m['customer']].str.lower().str.contains(search_q, na=False) | 
                        q_df[m['product']].str.lower().str.contains(search_q, na=False)]
        st.dataframe(q_df[[m['date'], m['country'], m['customer'], m['product'], m['qty'], m['amount'], '主要備註']], use_container_width=True)

else:
    st.info("💡 請管理員從左側側邊欄導入數據。")
