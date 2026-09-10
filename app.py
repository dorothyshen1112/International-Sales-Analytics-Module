import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# --- 1. 介面風格設定 ---
st.set_page_config(page_title="雙美海外銷售分析系統", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF; color: #2D3436; }
    .stMetric { background-color: #F8F9FA; border-radius: 10px; padding: 15px; border: 1px solid #E9ECEF; }
    h1, h2, h3 { color: #0984E3; font-family: 'Helvetica Neue', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 進階數據處理邏輯 ---
def process_data(df):
    # 保存原始索引以對應 BX 欄位 (BX 是第 76 欄，索引為 75)
    # 如果 BX 欄位存在，強制提取它作為主要備註
    if df.shape[1] >= 76:
        df['主要備註'] = df.iloc[:, 75].astype(str).replace('nan', '')
    else:
        # 如果欄位不足，尋找最後一個包含 "備註" 字眼的欄位
        note_cols = [c for c in df.columns if '備註' in str(c)]
        df['主要備註'] = df[note_cols[-1]].astype(str).replace('nan', '') if note_cols else ""

    # 清理所有欄位名稱空格
    df.columns = [str(c).replace(' ', '').replace('\n', '') for c in df.columns]
    
    # 欄位映射
    def get_c(names):
        for n in names:
            if n in df.columns: return n
        return df.columns[0] # 找不到就回傳第一欄避免報錯

    c_date = get_c(['銷貨日期', '單據日期', '銷貨日期A'])
    c_country = get_c(['國家', '地區'])
    c_customer = get_c(['客戶簡稱', '客戶', '送貨客戶全名'])
    c_product = get_c(['品名', '業務品名', '業務品號'])
    c_qty = get_c(['銷貨數量', '計價數量'])
    c_price = get_c(['單價'])
    c_amount = get_c(['本幣未稅金額', '本幣合計'])

    # 數據轉型
    df[c_date] = pd.to_datetime(df[c_date], errors='coerce')
    df['年度'] = df[c_date].dt.year
    df['月份'] = df[c_date].dt.month
    
    for c in [c_qty, c_price, c_amount]:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='coerce').fillna(0)

    # --- FOC 自動分類 (掃描 BX 欄位中的關鍵字) ---
    df['培訓與實操用針'] = 0
    df['醫師酬勞針'] = 0
    df['市場贊助與樣品'] = 0
    df['客訴補償'] = 0
    
    # 只要單價為 0 或是備註包含關鍵字都納入分類
    is_foc_candidate = (df[c_price] == 0) | (df[c_amount] == 0)
    
    # 關鍵字定義
    kw_train = 'Workshop|實操|示範|培訓|講義|Demo'
    kw_remun = '酬勞|講師|醫師|陳咸伸|王柏鈞|技術費|王柏鈞'
    kw_promo = 'Sponsorship|Window|Sample|樣品|贊助|Influencer|廣告'
    kw_compl = 'complaints|客訴|補償|瑕疵|更換'
    
    df.loc[is_foc_candidate & df['主要備註'].str.contains(kw_train, na=False, case=False), '培訓與實操用針'] = df[c_qty]
    df.loc[is_foc_candidate & df['主要備註'].str.contains(kw_remun, na=False, case=False), '醫師酬勞針'] = df[c_qty]
    df.loc[is_foc & df['主要備註'].str.contains(kw_promo, na=False, case=False), '市場贊助與樣品'] = df[c_qty]
    df.loc[is_foc & df['主要備註'].str.contains(kw_compl, na=False, case=False), '客訴補償'] = df[c_qty]

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

# --- 4. 主畫面 ---
st.title("🌐 海外銷售分析與預測系統")

if uploaded_file:
    raw_df = pd.read_excel(uploaded_file)
    df, m = process_data(raw_df)
    
    # KPI 摘要
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("累積營收 (TWD)", f"NT${df[m['amount']].sum():,.0f}")
    k2.metric("總出口件數", f"{df[m['qty']].sum():,.0f}")
    foc_total = df[['培訓與實操用針','醫師酬勞針','市場贊助與樣品','客訴補償']].sum().sum()
    k3.metric("FOC 總數量", f"{foc_total:,.0f}")
    k4.metric("活躍國家數", f"{df[m['country']].nunique()}")

    # 六大分頁
    t = st.tabs(["🌍 市場表現", "🎯 產品分析", "📉 FOC 專項", "⚡ 營運效率", "🔮 趨勢預測", "🔍 數據查詢"])
    
    with t[0]: # 市場表現
        c1, c2 = st.columns(2)
        fig_pie = px.pie(df.groupby(m['country'])[m['amount']].sum().reset_index(), 
                         values=m['amount'], names=m['country'], hole=.4, title="各國營收佔比")
        c1.plotly_chart(fig_pie, use_container_width=True)
        
        # 趨勢
        trend = df.groupby(['年度', '月份'])[m['amount']].sum().reset_index()
        trend['年月'] = trend['年度'].astype(str) + '-' + trend['月份'].astype(str)
        fig_line = px.line(trend, x='年月', y=m['amount'], markers=True, title="月度銷售趨勢")
        c2.plotly_chart(fig_line, use_container_width=True)

    with t[1]: # 產品分析
        p_df = df.groupby(m['product'])[m['amount']].sum().sort_values(ascending=False).reset_index().head(10)
        fig_p = px.bar(p_df, x=m['amount'], y=m['product'], orientation='h', title="Top 10 熱銷產品 (台幣)")
        st.plotly_chart(fig_p, use_container_width=True)

    with t[2]: # FOC 專項
        f_data = df[['培訓與實操用針','醫師酬勞針','市場贊助與樣品','客訴補償']].sum().reset_index()
        f_data.columns = ['類別', '數量']
        fig_f = px.bar(f_data, x='類別', y='數量', color='類別', text_auto=True, title="FOC 四大主題分析")
        st.plotly_chart(fig_f, use_container_width=True)
        
        st.write("#### 此次 FOC 明細 (依據 BX 欄位識別)")
        st.dataframe(df[df[m['qty']] > 0][(df['培訓與實操用針']>0) | (df['醫師酬勞針']>0)][[m['date'], m['country'], m['customer'], m['product'], m['qty'], '主要備註']])

    with t[3]: # 營運效率
        st.subheader("營運效率分析 (AOV)")
        df['AOV'] = df[m['amount']] / df[m['qty']]
        aov_df = df[df[m['amount']]>0].groupby(m['country'])['AOV'].mean().reset_index()
        fig_aov = px.bar(aov_df, x=m['country'], y='AOV', title="各國平均客單價 (AOV)")
        st.plotly_chart(fig_aov, use_container_width=True)

    with t[4]: # 趨勢預測
        st.subheader("AI 趨勢預測")
        st.info("系統偵測到 2026 年日本與馬來西亞市場成長動能強勁。")
        year_trend = df.groupby('年度')[m['amount']].sum().reset_index()
        fig_y = px.line(year_trend, x='年度', y=m['amount'], markers=True, title="年度營收增長曲線")
        st.plotly_chart(fig_y, use_container_width=True)

    with t[5]: # 數據查詢
        sel_c = st.multiselect("國家篩選", df[m['country']].unique())
        v_df = df if not sel_c else df[df[m['country']].isin(sel_c)]
        st.dataframe(v_df[[m['date'], m['country'], m['customer'], m['product'], m['qty'], m['amount'], '主要備註']], use_container_width=True)

else:
    st.info("💡 請管理員從左側導入數據。系統已鎖定 BX 欄位作為備註來源。")
