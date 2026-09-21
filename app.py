import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
import re

# --- 1. 頁面配置 (精密淨白風格) ---
st.set_page_config(page_title="双美全球銷售數據系統", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF; color: #2D3436; }
    .stMetric { background-color: #F8F9FA; border-radius: 10px; padding: 15px; border: 1px solid #E9ECEF; }
    [data-testid="stMetricValue"] { font-size: 1.6rem !important; color: #0984E3; font-weight: 700; }
    h1, h2, h3 { color: #0984E3; font-family: 'Helvetica Neue', sans-serif; }
    div.stTabs [data-baseweb="tab-list"] { gap: 15px; border-bottom: 2px solid #F1F3F5; }
    .report-note { 
        background-color: #E3F2FD; padding: 15px; border-left: 6px solid #2196F3; 
        border-radius: 4px; margin-bottom: 15px; font-size: 0.95rem; color: #1565C0; line-height: 1.6;
    }
    .highlight-text { color: #D32F2F; font-weight: bold; }
    .stDataFrame { overflow-x: auto !important; width: 100% !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 智慧數據處理引擎 ---
def smart_normalize(df):
    if df is None or df.empty or len(df.columns) < 2: return pd.DataFrame()
    
    # 預抓 BX 備註 (索引 75)
    bx_note = df.iloc[:, 75].astype(str).replace('nan', '') if df.shape[1] >= 76 else None
    
    # 清理標題
    df.columns = [str(c).strip().replace('\n', '').replace('\r', '').upper() for c in df.columns]
    
    m = {
        'DATE': ['銷貨日期', '單據日期', '日期', 'DATE', '銷貨日期A', '單據日', '成交日期', '開單日期'],
        'YEAR': ['年度', '年', 'YEAR', '年度_1', '年份', '西元年', 'YYYY', 'FISCALYEAR', '會計年度', '西元'],
        'COUNTRY': ['國家', '銷售地', '地區', '路線', '銷售地區', 'COUNTRY', '區域', '廠別名稱', '地  區', '省份', '市場', '國別', '銷貨地', '出貨地'],
        'CUSTOMER': ['客戶簡稱', '客戶', '客戶名稱', 'CUSTOMER', '送貨客戶全名', '經銷商', '代理商'],
        'PRODUCT': ['品名', '業務品名', '產品名稱', 'PRODUCT', '品號', '業務規格', '品  名', '項目名稱'],
        'QTY': ['銷貨數量', '數量', '計價數量', '總數量', 'QTY', '件數', '出貨數量'],
        'GIFT_QTY': ['贈/備品量', '贈品量', '贈品數量', 'GIFTQTY'],
        'PRICE': ['單價', '單價NT', 'PRICE', '單   價', '單  價', '單位成本'],
        'AMOUNT': ['本幣未稅金額', '本幣合計', '金額', '未稅金額', 'AMOUNT', '總計', '本幣金額', '銷貨金額', '幣別金額', '合計', '銷貨淨額', '未稅本幣']
    }

    def find_col(key):
        for name in m.get(key, []):
            for real_col in df.columns:
                if name == real_col or name in real_col: return real_col
        return None

    p_df = pd.DataFrame()
    
    # 時間與年度識別
    c_year = find_col('YEAR')
    c_date = find_col('DATE')
    if c_year:
        p_df['年度'] = pd.to_numeric(df[c_year].astype(str).str.extract(r'(\d{4})')[0], errors='coerce').fillna(0).astype(int)
        p_df['銷貨日期'] = pd.to_datetime(p_df['年度'].astype(str) + '-01-01', errors='coerce')
    elif c_date:
        dates = pd.to_datetime(df[c_date], errors='coerce')
        if dates.isna().all():
            p_df['年度'] = pd.to_numeric(df[c_date].astype(str).str.extract(r'(\d{4})')[0], errors='coerce').fillna(0).astype(int)
            p_df['銷貨日期'] = pd.to_datetime(p_df['年度'].astype(str) + '-01-01', errors='coerce')
        else:
            p_df['銷貨日期'] = dates
            p_df['年度'] = dates.dt.year.fillna(0).astype(int)
    else: return pd.DataFrame()

    p_df = p_df[(p_df['年度'] >= 2000) & (p_df['年度'] < 2040)].copy()
    if p_df.empty: return pd.DataFrame()
    p_df['月份'] = p_df['銷貨日期'].dt.month.fillna(1).astype(int)

    # 嚴格地點識別 (只看地點欄位，絕不讓「歐美皮膚科診所」等名稱影響國家判定)
    loc_cols = [c for c in df.columns if any(k in c for k in ['國家', '銷售地', '地區', '路線', '銷售地區', 'COUNTRY', '區域', '省份', '市場', '國別', '銷貨地', '出貨地'])]
    c_cust_col = find_col('CUSTOMER')
    raw_cust = df[c_cust_col].astype(str).loc[p_df.index] if c_cust_col else pd.Series(['']*len(p_df))

    def classify_strictly(idx):
        loc_vals = [str(df[c].loc[idx]).strip() for c in loc_cols if pd.notna(df[c].loc[idx]) and str(df[c].loc[idx]).strip() not in ['', 'nan', 'NAN', 'None', 'NONE', '0']]
        loc_raw = " ".join(loc_vals).upper()
        loc_clean = loc_raw.replace(' ', '').replace('　', '').replace('\t', '')
        
        cust_str = raw_cust.loc[idx].strip()
        cust_up = cust_str.upper()

        country = '其他'

        # 地點欄位優先判定
        if any(x in loc_clean for x in ['台灣', '臺灣', 'TAIWAN', 'TW']):
            country = '台灣'
        elif any(x in loc_clean for x in ['大陸', '中國', 'CHINA', 'MAINLAND', '北京', '上海', '廣州', '華東', '華南']):
            country = '大陸'
        elif any(x in loc_clean for x in ['德國', 'GERMANY', 'DEUTSCHLAND', 'DE']):
            country = '德國'
        elif any(x in loc_clean for x in ['歐美', '歐洲', '美洲', '美國', '北美', 'EUROPE', 'AMERICA', 'USA', 'EU']):
            country = '歐美'
        elif any(x in loc_clean for x in ['日本', 'JAPAN', 'JP']):
            country = '日本'
        elif any(x in loc_clean for x in ['新加坡', 'SINGAPORE', 'SG']):
            country = '新加坡'
        elif any(x in loc_clean for x in ['菲律賓', 'PHILIPPINES', 'PH']):
            country = '菲律賓'
        elif any(x in loc_clean for x in ['馬來西亞', 'MALAYSIA', 'MY']):
            country = '馬來西亞'
        elif any(x in loc_clean for x in ['泰國', 'THAILAND', 'TH']):
            country = '泰國'
        else:
            # 僅當地點欄位「完全為空」時，才容許從客戶名稱推測海外經銷商
            if 'VANGUARD' in cust_up:
                if 'OPC' in cust_up: country = '菲律賓'
                elif 'SDN' in cust_up: country = '馬來西亞'
                else: country = '新加坡'
            elif 'QUALTECH' in cust_up:
                country = '泰國'
            elif any(x in cust_up for x in ['東莞', '北京享贊']):
                country = '大陸'
            else:
                country = '台灣' if loc_clean == '' else '其他'

        # 模式與客戶全名歸類
        if country == '台灣':
            return '台灣', '台灣市場', '直營診所模式', cust_str if cust_str else '台灣診所'
        elif country == '日本':
            clean_clinic = re.sub(r'\s*(PTE\.?\s*LTD\.?|SDN\.?\s*BHD\.?|INC\.?|CO\.?|LTD\.?)$', '', cust_up).strip()
            return '日本', '海外市場', '直營診所模式', clean_clinic if clean_clinic else '日本診所'
        elif country == '大陸':
            if any(k in cust_up for k in ['東莞', 'DONGGUAN', '双美', '雙美']):
                return '大陸', '中國市場', '經銷商模式', '東莞双美'
            else:
                return '大陸', '中國市場', '經銷商模式', '北京享贊'
        elif country == '新加坡':
            return '新加坡', '海外市場', '經銷商模式', 'VANGUARD AESTHETICS PTE. LTD.'
        elif country == '菲律賓':
            return '菲律賓', '海外市場', '經銷商模式', 'VANGUARD AESTHETICS OPC'
        elif country == '馬來西亞':
            return '馬來西亞', '海外市場', '經銷商模式', 'Vanguard Aesthetics Sdn Bhd'
        elif country == '泰國':
            return '泰國', '海外市場', '經銷商模式', 'Qualtech Consulting (Thailand)'
        elif country == '德國':
            return '德國', '海外市場', '經銷商模式', cust_str if cust_str else '德國經銷商'
        elif country == '歐美':
            return '歐美', '海外市場', '經銷商模式', cust_str if cust_str else '歐美經銷商'
        else:
            return '其他', '海外市場', '經銷商模式', cust_str

    p_df['國家'], p_df['市場區域'], p_df['商務模式'], p_df['客戶'] = zip(*[classify_strictly(i) for i in p_df.index])

    # 數值轉換
    def to_num(key):
        col = find_col(key)
        if not col: return pd.Series([0.0]*len(p_df)).loc[p_df.index]
        return pd.to_numeric(df[col].loc[p_df.index].astype(str).str.replace(r'[^-0-9.]', '', regex=True), errors='coerce').fillna(0.0)

    p_df['產品'] = df[find_col('PRODUCT')].loc[p_df.index].astype(str).str.replace('Sunmax DeusaDerm', 'Sunmax Deusaderm VITAL', case=False).str.replace('VITAL VITAL', 'VITAL', case=False) if find_col('PRODUCT') else "未知產品"
    p_df['數量'] = to_num('QTY')
    p_df['贈品量'] = to_num('GIFT_QTY')
    p_df['單價'] = to_num('PRICE')
    p_df['金額'] = to_num('AMOUNT')
    p_df['備註'] = bx_note.loc[p_df.index].values if bx_note is not None else (df[find_col('NOTE')].loc[p_df.index].astype(str).replace('nan', '') if find_col('NOTE') else "")

    # FOC 計算
    p_df['實際FOC數量'] = np.where((p_df['金額'] <= 0) & (p_df['數量'] > 0), p_df['數量'], 0.0) + p_df['贈品量']
    p_df.loc[p_df['金額'] > 0, '實際FOC數量'] = 0.0
    
    foc_cats = ['培訓活動用針', '醫師酬勞', 'Workshop Training', 'Training/Rebate for JP', '市場贊助', '研究用針', '客訴補償', 'FOC樣品運輸']
    for cat in foc_cats: p_df[cat] = 0.0
    def classify_foc_row(row):
        f_qty = row['實際FOC數量']
        if f_qty <= 0: return '非FOC'
        rem = str(row['備註']).lower()
        if 'workshop training' in rem: return 'Workshop Training'
        if any(x in rem for x in ['實操針', '示範針', '會員實操針', '培訓針', 'demo', '施打用針', '教學']): return '培訓活動用針'
        if any(x in rem for x in ['酬勞針', '講師針', 'speech']): return '醫師酬勞'
        if any(x in rem for x in ['rebate', 'training']): return 'Training/Rebate for JP'
        if any(x in rem for x in ['sponsorship', 'launch']): return '市場贊助'
        if any(x in rem for x in ['sample needles', 'irb']): return '研究用針'
        if any(x in rem for x in ['complaints', '客訴', '補償']): return '客訴補償'
        return 'FOC樣品運輸'
    p_df['FOC類別'] = p_df.apply(classify_foc_row, axis=1)
    for cat in foc_cats: p_df[cat] = np.where(p_df['FOC類別'] == cat, p_df['實際FOC數量'], 0.0)
    p_df['FOC總量'] = p_df['實際FOC數量']
    p_df['收費量'] = np.where(p_df['金額'] > 0, p_df['數量'], 0.0)
    return p_df

# --- 3. 數據加載 ---
def load_all_data():
    if not os.path.exists("data.xlsx"): return None
    try:
        sheets = pd.read_excel("data.xlsx", sheet_name=None)
        all_dfs = [smart_normalize(df_sheet) for name, df_sheet in sheets.items() if not df_sheet.empty]
        return pd.concat(all_dfs, ignore_index=True) if all_dfs else None
    except Exception as e:
        st.error(f"數據讀取失敗: {e}")
        return None

# --- 4. 主畫面 ---
st.title("📊 双美全球銷售數據")
df_all = load_all_data()

if df_all is not None:
    sc1, sc2, sc3 = st.columns([1, 1, 2])
    sel_area = sc1.multiselect("區域", ['台灣市場', '中國市場', '海外市場'], default=['海外市場', '中國市場', '台灣市場'])
    available_countries = sorted(df_all[df_all['市場區域'].isin(sel_area)]['國家'].unique())
    sel_countries = sc3.multiselect("國家", available_countries, default=available_countries)
    all_yrs = sorted([int(y) for y in df_all['年度'].unique() if y > 0])
    sel_yrs = sc2.multiselect("年度", options=all_yrs, default=all_yrs)

    df = df_all[(df_all['年度'].isin(sel_yrs)) & (df_all['國家'].isin(sel_countries)) & (df_all['市場區域'].isin(sel_area))]

    if not df.empty:
        # KPI 卡片
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("總銷售金額", f"NT${df['金額'].sum():,.0f}")
        k2.metric("收費訂單量", f"{df['收費量'].sum():,.0f}")
        k3.metric("FOC 贈送總量", f"{df['FOC總量'].sum():,.0f}")
        tot_q = df['收費量'].sum() + df['FOC總量'].sum()
        f_rate = (df['FOC總量'].sum() / (tot_q + 0.0001)) * 100
        k4.metric("平均贈針比", f"{f_rate:.1f}%")
        k5.metric("活躍國家數", f"{df['國家'].nunique()}")

        tabs = st.tabs(["📊 市場佔比與YoY", "🎯 產品排行", "📉 FOC 專項分析", "⚡ 營運效率(進階)", "🔍 明細查詢"])
        
        # --- TAB 0: YoY 表格 ---
        with tabs[0]: 
            st.markdown("""<div class="report-note">
            <b>📊 市場佔比與 YoY 成長趨勢模組：</b><br>
            ● <b>分析重點：</b> 監控全球各市場營收佔比與年度擴張速度。YoY% 採用財務慣用之「紅漲綠跌」標註，新市場（去年數據為0）顯示為「-」。若特定市場連續兩年衰退，需啟動區域渠道檢討。
            </div>""", unsafe_allow_html=True)
            
            metric_opt = st.selectbox("分析指標", ["金額", "收費量", "FOC總量"])
            m_col = {'金額': '金額', '收費量': '收費量', 'FOC總量': 'FOC總量'}[metric_opt]
            st.plotly_chart(px.pie(df.groupby('國家')[m_col].sum().reset_index(), values=m_col, names='國家', hole=0.4, title="各國份額佔比", color_discrete_sequence=px.colors.qualitative.Pastel), use_container_width=True)
            st.divider()
            st.write("#### 📈 逐年成長趨勢表 (YoY%) - 欄位已放大，可向右滑動")
            
            pivot = df.pivot_table(index='國家', columns='年度', values=m_col, aggfunc='sum').fillna(0)
            sorted_y = sorted(pivot.columns)
            disp = pivot.copy()
            for i in range(1, len(sorted_y)):
                curr, prev = sorted_y[i], sorted_y[i-1]
                disp[f"{curr}年 成長%"] = np.where(pivot[prev]==0, np.nan, (pivot[curr]-pivot[prev])/pivot[prev]*100)
            
            final_f_cols = []
            col_cfg = {}
            format_dict = {}
            for y in sorted_y:
                y_c = f"{int(y)}年"
                disp = disp.rename(columns={y: y_c})
                final_f_cols.append(y_c)
                col_cfg[y_c] = st.column_config.NumberColumn(width=160, format="%,.0f")
                format_dict[y_c] = "{:,.0f}"
                
                g_c = f"{int(y)}年 成長%"
                if g_c in disp.columns:
                    final_f_cols.append(g_c)
                    col_cfg[g_c] = st.column_config.TextColumn(width=140)
                    format_dict[g_c] = lambda v: "-" if (pd.isna(v) or str(v).lower() in ['nan', 'none']) else f"{v:.1f}%"

            def color_logic(v, n):
                if '成長%' in str(n) and pd.notna(v) and str(v).lower() not in ['nan', 'none']:
                    try:
                        return 'color: #D32F2F; font-weight: bold' if float(v) > 0.001 else 'color: #388E3C; font-weight: bold'
                    except: return ''
                return ''
            
            st.dataframe(
                disp[final_f_cols].style.format(format_dict, na_rep="-").apply(lambda x: [color_logic(v, x.name) for v in x], axis=0),
                height=500, use_container_width=False, column_config=col_cfg
            )

        # --- TAB 1: 產品 ---
        with tabs[1]: 
            st.markdown("""<div class="report-note">
            <b>🎯 產品競爭力排行模組：</b><br>
            ● <b>分析重點：</b> 追蹤明星 SKU 與長青款走勢。監控高階利度卡因（LIDO 系列）在各市場的替換速度，並評估 VITAL 系列在成熟市場的銷量穩定度。
            </div>""", unsafe_allow_html=True)
            st.plotly_chart(px.bar(df.groupby('產品')['金額'].sum().sort_values(ascending=False).reset_index().head(12), x='金額', y='產品', orientation='h', color='金額', title="全球產品銷售排行榜"), use_container_width=True)

        # --- TAB 2: FOC ---
        with tabs[2]: 
            st.markdown("""<div class="report-note">
            <b>📉 FOC 贈針資源配置與成本模組：</b><br>
            ● <b>分析重點：</b><br>
            - <b>培訓活動用針</b>：市場教育投資，需比對受訓學員與診所後續的轉單訂購率。<br>
            - <b>醫師酬勞</b>：海外講師/KOL 技術交流成本，評估帶動的區域知名度與出貨拉動效應。<br>
            - <b>市場贊助與樣品</b>：新市場（如泰國、日本初期）拓展的獲客成本 (CAC)。<br>
            - <b>客訴補償</b>：產品與運送品質成本，若特定區域異常升高需檢驗冷鏈物流或防偽包裝。
            </div>""", unsafe_allow_html=True)
            f_cols_list = ['培訓活動用針', '醫師酬勞', 'Workshop Training', 'Training/Rebate for JP', '市場贊助', '研究用針', '客訴補償', 'FOC樣品運輸']
            st.plotly_chart(px.bar(df[f_cols_list].sum().reset_index().rename(columns={'index':'類別', 0:'數量'}), x='類別', y='數量', color='類別', text_auto=True), use_container_width=True)
            target = st.selectbox("🔍 FOC 明細過濾：", options=["全部 FOC"] + f_cols_list)
            f_data_view = df[df['FOC總量']>0].copy()
            if target != "全部 FOC": f_data_view = f_data_view[f_data_view['FOC類別'] == target]
            st.dataframe(f_data_view[['銷貨日期', '國家', '客戶', '產品', '數量', '贈品量', '金額', '備註']], use_container_width=True)

        # --- TAB 3: 營運效率 (5 大模組 + 藍色引航「分析重點」全數回歸！) ---
        with tabs[3]:
            st.subheader("⚡ 全球營運效率與模式深度解析")
            
            ce1, ce2 = st.columns(2)
            with ce1:
                st.markdown("""<div class="report-note">
                <b>1. 市場滲透度分析 (活躍客戶/診所數)：</b><br>
                ● <b>指標意義：</b> 日本與台灣為「直營診所模式」，數值代表實際開拓並下單的診所數量；其他經銷商國家（大陸 2 間、東南亞各 1 間、德國、歐美）代表合作經銷商數。<br>
                ● <b>分析重點：</b> 直營市場評估「診所拓點速度與黏著度」；經銷商市場監控「合作通路穩定度，若數值歸零代表代理合約存在中斷風險」。
                </div>""", unsafe_allow_html=True)
                active_c = df.groupby(['國家', '商務模式'])['客戶'].nunique().reset_index().sort_values('客戶', ascending=False)
                st.plotly_chart(px.bar(active_c, x='國家', y='客戶', color='商務模式', title="活躍客戶/診所總數", text_auto=True), use_container_width=True)
                with st.expander("📋 查看各國識別出的經銷商 / 診所名單"):
                    cust_dt = df.groupby(['國家', '商務模式'])['客戶'].unique().reset_index()
                    cust_dt['名單'] = cust_dt['客戶'].apply(lambda x: "\n".join([f"• {name}" for name in x]))
                    st.write(cust_dt[['國家', '商務模式', '名單']].to_html(escape=False).replace('\\n', '<br>'), unsafe_allow_html=True)

            with ce2:
                st.markdown("""<div class="report-note">
                <b>2. 物流模式分析 (平均單次訂單規模)：</b><br>
                ● <b>指標意義：</b> 經銷商大宗採購 (數值高)；直營診所小量多次進貨 (數值低)。<br>
                ● <b>分析重點：</b> 經銷商單筆採購若過低（如低於 100 支），代表頻繁小包報關，行政物流運費會嚴重侵蝕毛利；過高則需注意代理商庫存效期。
                </div>""", unsafe_allow_html=True)
                order_sz = df[df['收費量']>0].groupby(['國家', '商務模式']).agg({'收費量':'sum', '銷貨日期':'count'}).reset_index()
                order_sz['規模'] = (order_sz['收費量'] / (order_sz['銷貨日期'] + 0.0001)).round(1)
                st.plotly_chart(px.bar(order_sz.sort_values('規模', ascending=False), x='國家', y='規模', color='商務模式', title="平均單次訂單規模", text_auto=True), use_container_width=True)

            st.divider()
            ce3, ce4 = st.columns([2, 1])
            with ce3:
                st.markdown("""<div class="report-note">
                <b>3. 行銷投資回報 (FOC 轉換效率 / 槓桿比)：</b><br>
                ● <b>指標意義：</b> 每投入 1 支贈針(FOC)，平均帶動幾支收費訂單（評估行銷資源變現力）。<br>
                ● <b>分析重點：</b> 數值越高代表該國 Workshop、KOL 示範針的轉單能力越強；未投入贈針之純銷售市場（如大陸、歐美）不計入以避免極端值偏差。
                </div>""", unsafe_allow_html=True)
                
                eff_df = df.groupby('國家').agg({'收費量':'sum', 'FOC總量':'sum'}).reset_index()
                eff_plot = eff_df[eff_df['FOC總量'] > 0].copy()
                if not eff_plot.empty:
                    eff_plot['效率'] = (eff_plot['收費量'] / eff_plot['FOC總量']).round(1)
                    fig_eff = px.bar(eff_plot.sort_values('效率', ascending=False), x='國家', y='效率', 
                                     title="行銷槓桿比 (每投入1支FOC換回幾支收費針)", color='效率', text_auto=True)
                    st.plotly_chart(fig_eff, use_container_width=True)
                else:
                    st.info("💡 所選條件下無 FOC 贈針投入記錄。")
                
                with st.expander("📝 點此查看全市場 FOC 與收費量真實對照表"):
                    eff_df['槓桿比(倍)'] = np.where(eff_df['FOC總量'] > 0, (eff_df['收費量'] / eff_df['FOC總量']).round(1), 0.0)
                    st.dataframe(eff_df.rename(columns={'收費量':'收費量(支)', 'FOC總量':'贈針數(支)'}), use_container_width=True)

            with ce4:
                st.markdown("""<div class="report-note">
                <b>4. 全球商務模式佔比：</b><br>
                ● <b>指標意義：</b> 顯示全球營收中「直營診所模式」（台、日）與「代理經銷模式」（大陸、東南亞、歐美）的營收結構。<br>
                ● <b>分析重點：</b> 評估渠道抗風險能力。直營毛利高但管理重；經銷資金回籠快但話語權依賴度高。
                </div>""", unsafe_allow_html=True)
                st.plotly_chart(px.pie(df, names='商務模式', hole=0.5, color_discrete_sequence=['#0984E3', '#00B894'], title="全球模式營收佔比"), use_container_width=True)

            st.markdown("""<div class="report-note">
            <b>5. 全球採購季節性熱力圖 (Heatmap)：</b><br>
            ● <b>指標意義：</b> 以 1-12 月追蹤各國進貨淡旺季，顏色越深金額越高。<br>
            ● <b>分析重點：</b> 提早 2~3 個月掌握各國大型展會促銷後的補貨節奏，指導工廠排產、海外出差與庫存調度。
            </div>""", unsafe_allow_html=True)
            heat_df = df.groupby(['國家', '月份'])['金額'].sum().reset_index()
            fig_h = px.density_heatmap(heat_df, x='月份', y='國家', z='金額', color_continuous_scale='Blues', nbinsx=12, range_x=[0.5, 12.5], text_auto='.2s')
            fig_h.update_xaxes(tickmode='linear', tick0=1, dtick=1, ticktext=[f"{i}月" for i in range(1,13)], tickvals=list(range(1,13)))
            st.plotly_chart(fig_h, use_container_width=True)

        # --- TAB 4: 明細 ---
        with tabs[4]: 
            q = st.text_input("搜尋關鍵字...").lower()
            q_df = df.copy()
            if q: q_df = q_df[q_df['客戶'].str.lower().str.contains(q, na=False) | q_df['產品'].str.lower().str.contains(q, na=False)]
            st.dataframe(q_df[['銷貨日期', '市場區域', '商務模式', '國家', '客戶', '產品', '數量', '金額', '備註']], use_container_width=True)
else:
    st.warning("👋 尚未發現數據檔案。請確保 `data.xlsx` 已上傳。")
