import streamlit as st
import pandas as pd
import plotly.express as px

# --- 1. 介面風格設定 (精密淨白) ---
st.set_page_config(page_title="雙美海外銷售分析系統", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF; color: #2D3436; }
    .stMetric { background-color: #F8F9FA; border-radius: 10px; padding: 15px; border: 1px solid #E9ECEF; }
    h1, h2, h3 { color: #0984E3; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 數據處理邏輯 ---
def process_data(df):
    # 強制處理日期格式
    df['銷貨日期'] = pd.to_datetime(df['銷貨日期'], errors='coerce')
    df['年度'] = df['銷貨日期'].dt.year
    df['月份'] = df['銷貨日期'].dt.month
    
    # FOC 自動歸類欄位初始化
    for col in ['培訓與實操用針', '醫師酬勞針', '市場贊助與樣品', '客訴補償']:
        df[col] = 0
    
    # 自動分類邏輯 (單價為 0 或 折扣率為 0 的情況)
    is_foc = (df['單價'] == 0) | (df['折扣率 %'] == '0%') | (df['折扣率 %'] == 0)
    df.loc[is_foc & df['備註'].str.contains('實操|培訓|Workshop', na=False), '培訓與實操用針'] = df['銷貨數量']
    df.loc[is_foc & df['備註'].str.contains('酬勞|講師', na=False), '醫師酬勞針'] = df['銷貨數量']
    df.loc[is_foc & df['備註'].str.contains('贊助|Sponsorship|樣品|Sample', na=False), '市場贊助與樣品'] = df['銷貨數量']
    df.loc[is_foc & df['備註'].str.contains('客訴|補償|Complaints', na=False), '客訴補償'] = df['銷貨數量']
    return df

# --- 3. 側邊欄：管理員功能 ---
with st.sidebar:
    st.title("🛡️ 系統管理")
    admin_mode = st.toggle("管理員模式 (導入數據)")
    uploaded_file = None
    if admin_mode:
        pwd = st.text_input("輸入管理密碼", type="password")
        if pwd == "sunmax888":
            uploaded_file = st.file_uploader("上傳 ERP Excel", type=["xlsx"])
            if uploaded_file:
                st.success("數據導入成功！")

# --- 4. 主畫面分析 ---
st.title("🌐 海外銷售分析與預測 App")

if uploaded_file:
    # 讀取 Excel
    raw_df = pd.read_excel(uploaded_file)
    df = process_data(raw_df)
    
    # KPI 卡片
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("累積營收 (台幣)", f"NT${df['本幣未稅金額'].sum():,.0f}")
    m2.metric("總出口件數", f"{df['銷貨數量'].sum():,.0f}")
    m3.metric("FOC 總量", f"{df[['培訓與實操用針','醫師酬勞針','市場贊助與樣品','客訴補償']].sum().sum():,.0f}")
    m4.metric("活躍國家數", f"{df['國家'].nunique()}")

    # 六大模組分頁
    t1, t2, t3, t4 = st.tabs(["🌎 市場表現", "🎯 產品分析", "📉 FOC 追蹤", "🔍 數據查詢"])
    
    with t1:
        fig = px.pie(df.groupby('國家')['本幣未稅金額'].sum().reset_index(), 
                     values='本幣未稅金額', names='國家', hole=.4, title="各國台幣營收佔比")
        st.plotly_chart(fig, use_container_width=True)
        
    with t2:
        fig2 = px.bar(df.groupby('品名')['本幣未稅金額'].sum().sort_values(ascending=False).reset_index().head(10), 
                      x='本幣未稅金額', y='品名', orientation='h', title="前 10 大熱銷產品")
        st.plotly_chart(fig2, use_container_width=True)

    with t3:
        f_data = df[['培訓與實操用針','醫師酬勞針','市場贊助與樣品','客訴補償']].sum().reset_index()
        f_data.columns = ['類別', '數量']
        fig3 = px.bar(f_data, x='類別', y='數量', color='類別', text_auto=True, title="FOC 四大主題分析")
        st.plotly_chart(fig3, use_container_width=True)

    with t4:
        st.subheader("🔍 同仁自助查詢")
        sel_country = st.multiselect("選擇國家", df['國家'].unique())
        view_df = df if not sel_country else df[df['國家'].isin(sel_country)]
        st.dataframe(view_df[['銷貨日期','國家','客戶','品名','銷貨數量','本幣未稅金額','備註']], use_container_width=True)

else:
    st.info("💡 請管理員從左側開啟「管理員模式」並導入 Excel 數據以產生報表。")
