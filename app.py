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
    </style>
    """, unsafe_allow_html=True)

# --- 2. 智慧欄位匹配引擎 ---
def smart_process(df):
    # 先保存 BX 欄位(第76欄)作備案
    bx_note = df.iloc[:, 75].astype(str).replace('nan', '') if df.shape[1] >= 76 else None
    
    # 1. 清理欄位名稱 (去空格、去換行、轉大寫方便比對)
    raw_cols = df.columns.tolist()
    df.columns = [str(c).replace(' ', '').replace('\n', '').upper() for c in df.columns]
    
    # 2. 定義匹配字典 (讓新舊系統都能通)
    mapping_logic = {
        'DATE': ['銷貨日期', '單據日期', '日期', 'DATE', '銷貨日期A', '單據日'],
        'COUNTRY': ['國家', '地區', 'COUNTRY', '區域', '廠別名稱'],
        'CUSTOMER': ['客戶簡稱', '客戶', '客戶名稱', 'CUSTOMER', '送貨客戶全名'],
        'PRODUCT': ['品名', '業務品名', '產品名稱', 'PRODUCT', '業務規格', '品號'],
        'QTY': ['銷貨數量', '數量', '計價數量', '總數量', 'QTY', '件數'],
        'PRICE': ['單價', '單價NT', 'PRICE', '單價'],
        'AMOUNT': ['本幣未稅金額', '本幣合計', '金額', '未稅金額', 'AMOUNT', '總計', '本幣未稅金額_1'],
        'NOTE': ['備註', '備註1', 'REMARK', 'BX', '說明', '其他']
    }

    def find_best_col(targets):
        for t in targets:
            if t.upper() in df.columns: return t.upper()
        return None

    # 3. 執行智慧匹配
    c_map = {k: find_best_col(v) for k, v in mapping_logic.items()}
    
    # 4. 數據正規化
    # 日期
    if c_map['DATE']:
        df['DATE_CLEAN'] = pd.to_datetime(df[c_map['DATE']], errors='coerce')
    else:
        df['DATE_CLEAN'] = pd.Timestamp.now()
    
    df['年度'] = df['DATE_CLEAN'].dt.year
    df['月份'] = df['DATE_CLEAN'].dt.month
    
    # 數值
    for k in ['QTY', 'PRICE', 'AMOUNT']:
        col = c_map[k]
        if col:
            df[k] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        else:
            df[k] = 0
            
    # 國家 (若完全沒這欄，預設台灣)
    if c_map['COUNTRY']:
        df['國家_CLEAN'] = df[c_map['COUNTRY']].fillna('台灣')
    else:
        df['國家_CLEAN'] = '台灣'
        
    df['市場區域'] = df['國家_CLEAN'].apply(lambda x: '台灣市場' if '台灣' in str(x) else '海外市場')

    # 產品名稱 (合併 VITAL)
    prod_col = c_map['PRODUCT'] if c_map['PRODUCT'] else df.columns[0]
    df['產品_CLEAN'] = df[prod_col].astype(str).replace('Sunmax DeusaDerm', 'Sunmax Deusaderm VITAL')

    # 備註 (優先取第 BX 欄，其次找第一個出現的備註)
    if bx_note is not None:
        df['備註_CLEAN'] = bx_note
    elif c_map['NOTE']:
        df['備註_CLEAN'] = df[c_map['NOTE']].astype(str).replace('nan', '')
    else:
        df['備註_CLEAN'] = ""

    # 5. FOC 分類
    df['培訓與實操用針'] = 0
    df['醫師酬勞針'] = 0
    df['市場贊助與樣品'] = 0
    df['客訴補償'] = 0
    is_foc = (df['PRICE'] == 0) | (df['AMOUNT'] == 0)
    
    kw = {
        '培訓與實操用針': 'Workshop|實操|示範|培訓|講義|Demo',
        '醫師酬勞針': '酬勞|講師|醫師|陳咸伸|王柏鈞|技術費',
        '市場贊助與樣品': 'Sponsorship|Window|Sample|樣品|贊助|Influencer|廣告',
        '客訴補償': 'complaints|客訴|補償|瑕疵|更換|due to customer'
    }
    for label, pattern in kw.items():
        df.loc[is_foc & df['備註_CLEAN'].str.contains(pattern, na=False, case=False), label] = df['QTY']
    
    df['FOC總量'] = df[['培訓與實操用針', '醫師酬勞針', '市場贊助與樣品', '客訴補償']].sum(axis=1)
    df['收費量'] = np.where(df['AMOUNT'] > 0, df['QTY'], 0)

    return df

# --- 3. 側邊欄 ---
with st.sidebar:
    st.title("⚙️ 系統管理")
    admin_mode = st.toggle("管理員模式 (導入新/舊系統數據)")
    uploaded_file = None
    if admin_mode:
        pwd = st.text_input("管理密碼", type="password")
        if pwd == "sunmax888":
            uploaded_file = st.file_uploader("上傳 Excel (支援國內外/新舊系統)", type=["xlsx"])

st.title("🌐 雙美全球銷售數據中心")

if uploaded_file:
    raw_df = pd.read_excel(uploaded_file)
    df_all = smart_process(raw_df)
    
    # 全域篩選
    st.info("💡 智慧識別已完成：自動對接新舊系統欄位")
    fc1, fc2, fc3 = st.columns([1, 1, 2])
    sel_area = fc1.multiselect("區域", ['台灣市場', '海外市場'], default=['海外市場'])
    yrs = sorted(df_all['年度'].dropna().unique().astype(int))
    sel_yrs = fc2.multiselect("年度", yrs, default=yrs[-2:])
    countries = sorted(df_all[df_all['市場區域'].isin(sel_area)]['國家_CLEAN'].unique())
    sel_countries = fc3.multiselect("國家", countries, default=countries)

    df = df_all[(df_all['年度'].isin(sel_yrs)) & (df_all['國家_CLEAN'].isin(sel_countries)) & (df_all['市場區域'].isin(sel_area))]

    if not df.empty:
        # KPI
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("所選營收", f"NT${df['AMOUNT'].sum():,.0f}")
        k2.metric("收費量", f"{df['收費量'].sum():,.0f}")
        k3.metric("FOC 總量", f"{df['FOC總量'].sum():,.0f}")
        k4.metric("FOC 佔比", f"{(df['FOC總量'].sum()/(df['QTY'].sum()+0.001)*100):.1%}%")
        k5.metric("國家數", f"{df['國家_CLEAN'].nunique()}")

        tabs = st.tabs(["📊 市場佔比與成長", "🎯 產品排行", "📉 FOC 專項", "⚡ 營運效率", "🔍 數據查詢"])

        with tabs[0]: # 市場佔比
            compare_metric = st.selectbox("分析指標", ["銷售金額", "收費量", "FOC 總量"])
            m_col = {'銷售金額': 'AMOUNT', '收費量': '收費量', 'FOC 總量': 'FOC總量'}[compare_metric]
            
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.pie(df.groupby('國家_CLEAN')[m_col].sum().reset_index(), values=m_col, names='國家_CLEAN', hole=0.4, title="全球市場份額"), use_container_width=True)
            
            pivot_df = df.pivot_table(index='國家_CLEAN', columns='年度', values=m_col, aggfunc='sum').fillna(0)
            for i in range(1, len(pivot_df.columns)):
                curr, prev = pivot_df.columns[i], pivot_df.columns[i-1]
                pivot_df[f"{curr} YoY%"] = ((pivot_df[curr] - pivot_df[prev]) / (pivot_df[prev] + 0.001) * 100).replace([np.inf, -np.inf], 0)
            st.dataframe(pivot_df.style.format("{:,.1f}"))

        with tabs[1]: # 產品分析
            st.plotly_chart(px.bar(df.groupby('產品_CLEAN')['AMOUNT'].sum().sort_values(ascending=False).reset_index().head(15), x='AMOUNT', y='產品_CLEAN', orientation='h', title="產品營收 Top 15 (全球)"), use_container_width=True)

        with tabs[2]: # FOC
            f_sum = df[['培訓與實操用針', '醫師酬勞針', '市場贊助與樣品', '客訴補償']].sum().reset_index()
            f_sum.columns = ['類別', '數量']
            st.plotly_chart(px.bar(f_sum, x='類別', y='數量', color='類別', text_auto=True), use_container_width=True)
            st.dataframe(df[df['FOC總量']>0][['DATE_CLEAN', '國家_CLEAN', '產品_CLEAN', 'QTY', '備註_CLEAN']], use_container_width=True)

        with tabs[3]: # 營運效率
            st.write("#### 台灣 vs 海外：成長引擎分析")
            st.plotly_chart(px.bar(df.groupby(['年度', '市場區域'])['AMOUNT'].sum().reset_index(), x='年度', y='AMOUNT', color='市場區域', barmode='group'), use_container_width=True)
            eff_df = df.groupby('國家_CLEAN').agg({'收費量':'sum', 'FOC總量':'sum'}).reset_index()
            eff_df['轉換率'] = eff_df['收費量'] / (eff_df['FOC總量'] + 0.001)
            st.plotly_chart(px.bar(eff_df.sort_values('轉換率', ascending=False), x='國家_CLEAN', y='轉換率', title="FOC 投資轉換率 (越高越好)", color='轉換率'), use_container_width=True)

        with tabs[4]: # 查詢
            st.dataframe(df[['DATE_CLEAN', '市場區域', '國家_CLEAN', '產品_CLEAN', 'QTY', 'AMOUNT', '備註_CLEAN']], use_container_width=True)

else:
    st.info("👋 請管理員上傳 Excel (支援包含「台灣」或「舊系統」之數據)。系統會自動智慧匹配欄位名稱！")
