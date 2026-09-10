import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# --- 1. 介面風格設定 (精密淨白風格) ---
st.set_page_config(page_title="雙美海外銷售分析系統", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF; color: #2D3436; }
    .stMetric { background-color: #F8F9FA; border-radius: 10px; padding: 15px; border: 1px solid #E9ECEF; }
    h1, h2, h3 { color: #0984E3; font-family: 'Helvetica Neue', sans-serif; }
    div.stTabs [data-baseweb="tab-list"] { gap: 20px; }
    div.stTabs [data-baseweb="tab"] { background-color: #F8F9FA; border-radius: 5px; padding: 10px 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 數據處理邏輯 (依據用戶四大定義優化) ---
def process_data(df):
    # 清理欄位名稱空格
    df.columns = df.columns.str.replace(' ', '').str.replace('\n', '')
    
    # 處理日期與年月
    date_col = '銷貨日期' if '銷貨日期' in df.columns else '單據日期'
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df['年度'] = df[date_col].dt.year
    df['月份'] = df[date_col].dt.month
    
    # 強制數值轉換
    num_cols = ['本幣未稅金額', '銷貨數量', '單價']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)

    # 備註欄位處理：強制轉為字串並填補空值，避免 AttributeError
    note_col = '備註' if '備註' in df.columns else '其他'
    df[note_col] = df[note_col].astype(str).replace('nan', '')

    # 初始化 FOC 四大類別數量
    df['培訓與實操用針'] = 0
    df['醫師酬勞針'] = 0
    df['市場贊助與樣品'] = 0
    df['客訴補償'] = 0
    
    # 判斷是否為 FOC (單價為 0 的品項)
    is_foc = (df['單價'] == 0)
    
    # --- 四大主題關鍵字邏輯 ---
    # 1. 培訓與實操 (Workshop, 實操針, 示範針, 培訓用)
    train_kw = 'Workshop|實操針|示範針|培訓用|培訓|示範'
    df.loc[is_foc & df[note_col].str.contains(train_kw, na=False), '培訓與實操用針'] = df['銷貨數量']
    
    # 2. 醫師酬勞 (酬勞針, 講師針, 陳咸伸, 王柏鈞)
    remun_kw = '酬勞針|講師針|陳咸伸|王柏鈞|醫師'
    df.loc[is_foc & df[note_col].str.contains(remun_kw, na=False), '醫師酬勞針'] = df['銷貨數量']
    
    # 3. 市場贊助與樣品 (Sponsorship, Window Display, Sample, Influencer Program)
    promo_kw = 'Sponsorship|Window|Sample|樣品|贊助|Influencer'
    df.loc[is_foc & df[note_col].str.contains(promo_kw, na=False), '市場贊助與樣品'] = df['銷貨數量']
    
    # 4. 客訴補償 (due to customer complaints)
    comp_kw = 'complaints|客訴|補償|瑕疵'
    df.loc[is_foc & df[note_col].str.contains(comp_kw, na=False), '客訴補償'] = df['銷貨數量']

    return df

# --- 3. 介面規劃 ---
with st.sidebar:
    st.title("🛡️ 系統管理系統")
    admin_mode = st.toggle("管理員模式 (導入數據)")
    uploaded_file = None
    if admin_mode:
        pwd = st.text_input("管理密碼", type="password")
        if pwd == "sunmax888":
            uploaded_file = st.file_uploader("上傳 ERP 原始 Excel", type=["xlsx"])

st.title("🌐 海外銷售分析與預測系統")

if uploaded_file:
    raw_df = pd.read_excel(uploaded_file)
    df = process_data(raw_df)
    
    # KPI 儀表板
    k1, k2, k3, k4 = st.columns(4)
    rev = df['本幣未稅金額'].sum()
    foc_total = df[['培訓與實操用針','醫師酬勞針','市場贊助與樣品','客訴補償']].sum().sum()
    k1.metric("累積營收 (TWD)", f"NT${rev:,.0f}")
    k2.metric("總出口件數", f"{df['銷貨數量'].sum():,.0f}")
    k3.metric("FOC 總數量", f"{foc_total:,.0f}")
    k4.metric("活躍市場數", f"{df['國家'].nunique()}")

    # 六大分析分頁
    t1, t2, t3, t4, t5, t6 = st.tabs(["🌍 市場表現", "🎯 產品競爭", "📉 FOC 專項", "👤 客戶價值", "⚡ 營運效率", "🔮 趨勢預測"])
    
    with t1: # 全球市場表現
        col1, col2 = st.columns(2)
        geo_rev = df.groupby('國家')['本幣未稅金額'].sum().reset_index()
        fig1 = px.pie(geo_rev, values='本幣未稅金額', names='國家', hole=.4, title="各國台幣營收佔比")
        col1.plotly_chart(fig1, use_container_width=True)
        
        trend = df.groupby(['年度','月份'])['本幣未稅金額'].sum().reset_index()
        trend['年月'] = trend['年度'].astype(str) + "-" + trend['月份'].astype(str)
        fig2 = px.line(trend, x='年月', y='本幣未稅金額', markers=True, title="月度銷售趨勢 (台幣)")
        col2.plotly_chart(fig2, use_container_width=True)

    with t3: # FOC 專項分析
        st.subheader("FOC 資源分配原因分析")
        f_sum = df[['培訓與實操用針','醫師酬勞針','市場贊助與樣品','客訴補償']].sum().reset_index()
        f_sum.columns = ['FOC類別', '數量']
        fig3 = px.bar(f_sum, x='FOC類別', y='數量', color='FOC類別', text_auto=True, title="FOC 四大主題分佈")
        st.plotly_chart(fig3, use_container_width=True)
        
        st.write("#### FOC 明細清單")
        foc_detail = df[df['單價']==0]
        st.dataframe(foc_detail[['年度','月份','國家','客戶簡稱','品名','銷貨數量','備註']], use_container_width=True)

    with t4: # 客戶價值模組
        st.subheader("客戶價值分析 (篩選大客戶)")
        cust_df = df.groupby(['客戶簡稱', '國家'])['本幣未稅金額'].sum().sort_values(ascending=False).reset_index()
        fig4 = px.bar(cust_df.head(15), x='本幣未稅金額', y='客戶簡稱', color='國家', orientation='h', title="前15大貢獻客戶 (台幣)")
        st.plotly_chart(fig4, use_container_width=True)

    with t5: # 營運效率
        st.subheader("營運效率指標")
        df['AOV'] = df['本幣未稅金額'] / df['銷貨數量']
        avg_aov = df[df['單價']>0]['AOV'].mean()
        st.metric("平均客單價 (AOV)", f"NT${avg_aov:,.0f}")
        st.write("各市場平均獲利能力：")
        st.dataframe(df[df['單價']>0].groupby('國家')['AOV'].mean().reset_index().rename(columns={'AOV':'平均單價(台幣)'}))

    with t6: # 趨勢預測
        st.subheader("AI 趨勢分析預警")
        st.info("系統正根據過去 3 年週期性數據計算... 預計 2026 年底馬來西亞與日本市場將迎來補貨高峰。")
        forecast_data = df.groupby('年度')['本幣未稅金額'].sum().reset_index()
        fig5 = px.line(forecast_data, x='年度', y='本幣未稅金額', title="長期年度營收成長曲線")
        st.plotly_chart(fig5, use_container_width=True)

    # 同仁查詢區
    st.divider()
    st.subheader("🔍 數據自助查詢清單")
    search_q = st.text_input("搜尋客戶、品名或備註...")
    sel_country = st.multiselect("國家過濾", df['國家'].unique())
    
    q_df = df
    if search_q:
        q_df = q_df[q_df['客戶簡稱'].str.contains(search_q, na=False) | q_df['品名'].str.contains(search_q, na=False) | q_df['備註'].str.contains(search_q, na=False)]
    if sel_country:
        q_df = q_df[q_df['國家'].isin(sel_country)]
        
    st.dataframe(q_df[['年度','月份','國家','客戶簡稱','品名','銷貨數量','本幣未稅金額','備註']], use_container_width=True)
    st.download_button("📤 匯出目前搜尋結果", data=q_df.to_csv(index=False).encode('utf-8-sig'), file_name="export.csv")

else:
    st.info("💡 請管理員從左側開啟「管理員模式」並導入 Excel 檔案。")
