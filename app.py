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

# --- 2. 核心數據清洗邏輯 (解決重複欄位與備註問題) ---
def process_data(df):
    # 第一步：處理重複欄位名稱 (ERP 常見問題)
    cols = []
    count = {}
    for col in df.columns:
        clean_name = str(col).replace(' ', '').replace('\n', '')
        if clean_name in count:
            count[clean_name] += 1
            cols.append(f"{clean_name}_{count[clean_name]}")
        else:
            count[clean_name] = 0
            cols.append(clean_name)
    df.columns = cols

    # 第二步：定義我們需要的關鍵欄位映射
    def find_col(possible_names):
        for name in possible_names:
            if name in df.columns: return name
        return None

    c_date = find_col(['銷貨日期', '單據日期', '銷貨日期A'])
    c_country = find_col(['國家', '地區'])
    c_customer = find_col(['客戶簡稱', '客戶', '送貨客戶全名'])
    c_product = find_col(['品名', '業務品名', '品號'])
    c_qty = find_col(['銷貨數量', '總數量'])
    c_price = find_col(['單價'])
    c_amount = find_col(['本幣未稅金額', '本幣合計'])
    # 針對備註，抓取第一個出現的備註欄位
    c_note = find_col(['備註', '備註_1', '其他'])

    # 第三步：數據轉換
    df[c_date] = pd.to_datetime(df[c_date], errors='coerce')
    df['年度'] = df[c_date].dt.year
    df['月份'] = df[c_date].dt.month
    
    # 數值清理
    for c in [c_qty, c_price, c_amount]:
        if c:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='coerce').fillna(0)

    # 備註清理 (確保是字串格式)
    df['CleanNote'] = df[c_note].astype(str).replace('nan', '')

    # 第四步：FOC 自動分類 (依照四大主題)
    df['培訓與實操用針'] = 0
    df['醫師酬勞針'] = 0
    df['市場贊助與樣品'] = 0
    df['客訴補償'] = 0
    
    is_foc = (df[c_price] == 0)
    
    # 關鍵字定義
    kw_train = 'Workshop|實操|示範|培訓|講義|Demo'
    kw_remun = '酬勞|講師|醫師|陳咸伸|王柏鈞|技術費'
    kw_promo = 'Sponsorship|Window|Sample|樣品|贊助|Influencer|廣告'
    kw_compl = 'complaints|客訴|補償|瑕疵|更換'
    
    df.loc[is_foc & df['CleanNote'].str.contains(kw_train, na=False, case=False), '培訓與實操用針'] = df[c_qty]
    df.loc[is_foc & df['CleanNote'].str.contains(kw_remun, na=False, case=False), '醫師酬勞針'] = df[c_qty]
    df.loc[is_foc & df['CleanNote'].str.contains(kw_promo, na=False, case=False), '市場贊助與樣品'] = df[c_qty]
    df.loc[is_foc & df['CleanNote'].str.contains(kw_compl, na=False, case=False), '客訴補償'] = df[c_qty]

    # 回傳整理後的結果
    return df, {
        'date': c_date, 'country': c_country, 'customer': c_customer,
        'product': c_product, 'qty': c_qty, 'amount': c_amount, 'note': c_note
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
    df, mapping = process_data(raw_df)
    
    # KPI
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("累積營收 (TWD)", f"NT${df[mapping['amount']].sum():,.0f}")
    k2.metric("總出口件數", f"{df[mapping['qty']].sum():,.0f}")
    foc_sum = df[['培訓與實操用針','醫師酬勞針','市場贊助與樣品','客訴補償']].sum().sum()
    k3.metric("FOC 總數量", f"{foc_sum:,.0f}")
    k4.metric("活躍國家數", f"{df[mapping['country']].nunique()}")

    # Tabs
    t1, t2, t3, t4 = st.tabs(["🌍 市場表現", "🎯 產品分析", "📉 FOC 專項", "🔍 數據查詢"])
    
    with t1:
        fig1 = px.pie(df.groupby(mapping['country'])[mapping['amount']].sum().reset_index(), 
                     values=mapping['amount'], names=mapping['country'], hole=.4, title="各國台幣營收佔比")
        st.plotly_chart(fig1, use_container_width=True)
        
    with t3:
        f_summary = df[['培訓與實操用針','醫師酬勞針','市場贊助與樣品','客訴補償']].sum().reset_index()
        f_summary.columns = ['類別', '數量']
        fig3 = px.bar(f_summary, x='類別', y='數量', color='類別', text_auto=True, title="FOC 四大主題分析")
        st.plotly_chart(fig3, use_container_width=True)

    with t4:
        st.subheader("🔍 同仁自助查詢")
        sel_c = st.multiselect("國家篩選", df[mapping['country']].unique())
        v_df = df if not sel_c else df[df[mapping['country']].isin(sel_c)]
        display_cols = [mapping['date'], mapping['country'], mapping['customer'], mapping['product'], mapping['qty'], mapping['amount'], mapping['note']]
        st.dataframe(v_df[display_cols], use_container_width=True)

else:
    st.info("💡 請管理員從左側導入數據。系統已自動相容 ERP 重複欄位格式。")
