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
    h1, h2, h3 { color: #0984E3; font-family: 'Helvetica Neue', sans-serif; }
    div.stTabs [data-baseweb="tab-list"] { gap: 15px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 核心數據處理 ---
def process_data(df):
    # 抓取第 76 欄作為主要備註 (BX欄位)
    if df.shape[1] >= 76:
        raw_note = df.iloc[:, 75].astype(str).replace('nan', '')
    else:
        raw_note = df.iloc[:, -1].astype(str).replace('nan', '')

    # 清理欄位名稱
    df.columns = [str(c).replace(' ', '').replace('\n', '') for c in df.columns]
    df['主要備註'] = raw_note

    # 欄位映射自動匹配
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

    # 轉型與清理
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
    is_foc = (df[c_price] == 0) | (df[c_amount] == 0)
    
    kw_dict = {
        '培訓與實操用針': 'Workshop|實操|示範|培訓|講義|Demo',
        '醫師酬勞針': '酬勞|講師|醫師|陳咸伸|王柏鈞|技術費',
        '市場贊助與樣品': 'Sponsorship|Window|Sample|樣品|贊助|Influencer|廣告',
        '客訴補償': 'complaints|客訴|補償|瑕疵|更換|due to customer'
    }
    
    for key, kw in kw_dict.items():
        df.loc[is_foc & df['主要備註'].str.contains(kw, na=False, case=False), key] = df[c_qty]

    return df, {
        'date': c_date, 'country': c_country, 'customer': c_customer,
        'product': c_product, 'qty': c_qty, 'amount': c_amount, 'note': '主要備註'
    }

# --- 3. 側邊欄導入 ---
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
    
    # KPI 摘要
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("累積營收 (TWD)", f"NT${df[m['amount']].sum():,.0f}")
    k2.metric("總出口件數", f"{df[m['qty']].sum():,.0f}")
    foc_cols = ['培訓與實操用針', '醫師酬勞針', '市場贊助與樣品', '客訴補償']
    k3.metric("FOC 總數量", f"{df[foc_cols].sum().sum():,.0f}")
    k4.metric("活躍國家數", f"{df[m['country']].nunique()}")

    # --- 核心分頁 ---
    tabs = st.tabs(["🌍 市場對比分析", "🎯 產品分析", "📉 FOC 追蹤", "👤 客戶價值", "⚡ 營運效率", "🔍 數據查詢"])
    
    with tabs[0]: # 修改為強大的「跨維度對比模組」
        st.subheader("📈 跨年度與市場綜合對比")
        
        # 篩選控制項
        c1, c2, c3 = st.columns([1, 1, 2])
        compare_metric = c1.selectbox("選擇分析指標", ["銷售金額 (台幣)", "出口數量"], index=0)
        target_col = m['amount'] if compare_metric == "銷售金額 (台幣)" else m['qty']
        
        selected_years = c2.multiselect("選擇年度 (可多選進行比較)", options=sorted(df['年度'].dropna().unique()), default=df['年度'].dropna().unique()[-2:])
        selected_countries = c3.multiselect("選擇國家", options=df[m['country']].unique(), default=df[m['country']].unique()[:3])
        
        # 數據過濾
        comp_df = df[(df['年度'].isin(selected_years)) & (df[m['country']].isin(selected_countries))]
        
        if not comp_df.empty:
            # 1. 柱狀對比圖
            fig_compare = px.bar(comp_df.groupby(['年度', m['country']])[target_col].sum().reset_index(), 
                                 x=m['country'], y=target_col, color='年度', barmode='group',
                                 text_auto='.2s', title=f"{compare_metric} - 年度對比圖")
            st.plotly_chart(fig_compare, use_container_width=True)
            
            # 2. 數據表格與成長率計算
            st.write("#### 數據摘要與 YoY 成長率分析")
            pivot_df = comp_df.pivot_table(index=m['country'], columns='年度', values=target_col, aggfunc='sum').fillna(0)
            
            # 計算成長率 (如果至少有兩年數據)
            if len(selected_years) >= 2:
                latest_yr = max(selected_years)
                prev_yr = sorted(selected_years)[-2]
                pivot_df['成長率 (%)'] = ((pivot_df[latest_yr] - pivot_df[prev_yr]) / pivot_df[prev_yr] * 100).replace([np.inf, -np.inf], 0).fillna(0)
            
            st.table(pivot_df.style.format("{:,.0f}").background_gradient(cmap='Blues'))

    with tabs[1]: # 產品分析
        st.subheader("產品熱銷排名 (台幣金額)")
        p_df = df[df[m['amount']] > 0].groupby(m['product'])[m['amount']].sum().sort_values(ascending=False).reset_index().head(12)
        fig_p = px.bar(p_df, x=m['amount'], y=m['product'], orientation='h', color=m['product'], title="熱銷產品 Top 12")
        st.plotly_chart(fig_p, use_container_width=True)

    with tabs[2]: # FOC 專項
        st.subheader("FOC 資源配置原因 (BX欄位識別)")
        f_summary = df[foc_cols].sum().reset_index()
        f_summary.columns = ['類別', '數量']
        fig_f = px.bar(f_summary, x='類別', y='數量', color='類別', text_auto=True)
        st.plotly_chart(fig_f, use_container_width=True)
        st.dataframe(df[df[foc_cols].sum(axis=1) > 0][[m['date'], m['country'], m['customer'], m['product'], m['qty'], '主要備註']], use_container_width=True)

    with tabs[4]: # 營運效率 (AOV)
        st.subheader("營運效率指標 (平均客單價)")
        df['AOV'] = df[m['amount']] / df[m['qty']]
        aov_data = df[df[m['amount']]>0].groupby(m['country'])['AOV'].mean().reset_index()
        fig_aov = px.bar(aov_data, x=m['country'], y='AOV', title="各國平均每件針劑台幣價值")
        st.plotly_chart(fig_aov, use_container_width=True)

    with tabs[5]: # 查詢
        st.subheader("🔍 同仁自助查詢")
        search_q = st.text_input("輸入關鍵字搜尋...")
        q_df = df
        if search_q:
            q_df = q_df[q_df[m['customer']].str.contains(search_q, na=False) | q_df[m['product']].str.contains(search_q, na=False)]
        st.dataframe(q_df[[m['date'], m['country'], m['customer'], m['product'], m['qty'], m['amount'], '主要備註']], use_container_width=True)

else:
    st.info("💡 請管理員從左側側邊欄導入 ERP Excel 數據以開啟分析。")
