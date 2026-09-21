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
    
    # 清理標題：去空格、換行、轉大寫
    df.columns = [str(c).strip().replace('\n', '').replace('\r', '').upper() for c in df.columns]
    
    m = {
        'DATE': ['銷貨日期', '單據日期', '日期', 'DATE', '銷貨日期A', '單據日', '成交日期', '開單日期'],
        'YEAR': ['年度', '年', 'YEAR', '年度_1', '年份', '西元年', 'YYYY', 'FISCALYEAR', '會計年度', '西元'],
        'COUNTRY': ['國家', '銷售地', '地區', '路線', '銷售地區', 'COUNTRY', '區域', '廠別名稱', '地  區', '省份', '市場', '國別', '銷貨地', '出貨地', '客戶地址'],
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

    # 找出所有地點欄位與客戶欄位
    loc_cols = [c for c in df.columns if any(k in c for k in ['國家', '銷售地', '地區', '路線', 'COUNTRY', '區域', '省份', '市場', '國別'])]
    c_cust_col = find_col('CUSTOMER')
    raw_cust = df[c_cust_col].astype(str).loc[p_df.index] if c_cust_col else pd.Series(['']*len(p_df))

    # --- 關鍵核心：徹底消除全半形空格後再分類，解決「歐 美」掉入「其他」問題 ---
    def classify_all(idx):
        loc_text_raw = " ".join([str(df[c].loc[idx]) for c in loc_cols if str(df[c].loc[idx]) != 'nan'])
        cust_raw = raw_cust.loc[idx].strip()
        
        full_info = (loc_text_raw + " " + cust_raw).upper()
        # 移除所有半形與全形空格，讓「歐 美」、「德 國」無所遁形
        compact = full_info.replace(' ', '').replace('　', '').replace('\t', '')

        # 1. 德國 (獨立分類)
        if any(x in compact for x in ['德國', 'GERMANY', 'DEUTSCHLAND']) or ' DE ' in (' ' + full_info + ' '):
            return '德國', '海外市場', '經銷商模式', cust_raw if cust_raw else '德國經銷商'

        # 2. 歐美 (徹底抓取「歐美」、「歐美線」、「美國」、「歐洲」、「USA」)
        if any(x in compact for x in ['歐美', '歐洲', '美洲', '美國', '北美', 'EUROPE', 'AMERICA', 'USA']) or any(x in full_info.split() for x in ['US', 'USA', 'EU']):
            return '歐美', '海外市場', '經銷商模式', cust_raw if cust_raw else '歐美經銷商'

        # 3. 日本 (直營診所模式)
        if any(x in compact for x in ['日本', 'JAPAN']) or ' JP ' in (' ' + full_info + ' '):
            clean_clinic = re.sub(r'\s*(PTE\.?\s*LTD\.?|SDN\.?\s*BHD\.?|INC\.?|CO\.?|LTD\.?)$', '', cust_raw, flags=re.IGNORECASE).strip()
            return '日本', '海外市場', '直營診所模式', clean_clinic if clean_clinic else '日本診所'

        # 4. 新加坡 (Vanguard)
        if any(x in compact for x in ['新加坡', 'SINGAPORE']) or ('VANGUARD' in compact and not any(k in compact for k in ['MALAYSIA', 'PHILIPPINES', '馬來西亞', '菲律賓'])):
            return '新加坡', '海外市場', '經銷商模式', 'VANGUARD AESTHETICS PTE. LTD.'

        # 5. 菲律賓 (Vanguard)
        if any(x in compact for x in ['菲律賓', 'PHILIPPINES', 'OPC']) or ' PH ' in (' ' + full_info + ' '):
            return '菲律賓', '海外市場', '經銷商模式', 'VANGUARD AESTHETICS OPC'

        # 6. 馬來西亞 (Vanguard)
        if any(x in compact for x in ['馬來西亞', 'MALAYSIA', 'SDNBHD']) or ' MY ' in (' ' + full_info + ' '):
            return '馬來西亞', '海外市場', '經銷商模式', 'Vanguard Aesthetics Sdn Bhd'

        # 7. 泰國 (Qualtech)
        if any(x in compact for x in ['泰國', 'THAILAND', 'QUALTECH']) or ' TH ' in (' ' + full_info + ' '):
            return '泰國', '海外市場', '經銷商模式', 'Qualtech Consulting (Thailand)'

        # 8. 台灣 (直營診所模式)
        if any(x in compact for x in ['台灣', '臺灣', 'TAIWAN']) or ' TW ' in (' ' + full_info + ' '):
            return '台灣', '台灣市場', '直營診所模式', cust_raw if cust_raw else '台灣診所'

        # 9. 大陸 (經銷商模式：分為 東莞双美 與 北京享贊 兩間)
        if any(x in compact for x in ['大陸', '中國', 'CHINA', '北京', '上海', '廣州', '華東', '華南', '東莞', '享贊', '享赞']):
            if any(x in compact for x in ['東莞', 'DONGGUAN', '双美', '雙美']):
                return '大陸', '中國市場', '經銷商模式', '東莞双美'
            else:
                return '大陸', '中國市場', '經銷商模式', '北京享贊'

        # 10. 若地點欄位本身有寫明確文字，直接提取真實名稱，避免變成「其他」
        for c in loc_cols:
            val = str(df[c].loc[idx]).strip()
            if val and val.lower() not in ['nan', 'none', '', '0']:
                clean_v = val.replace(' ', '').replace('　', '')
                return clean_v, '海外市場', '經銷商模式', cust_raw if cust_raw else clean_v

        return '其他', '海外市場', '經銷商模式', cust_raw if cust_raw else '未分類客戶'

    p_df['國家'], p_df['市場區域'], p_df['商務模式'], p_df['客戶'] = zip(*[classify_all(i) for i in p_df.index])

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
            st.plotly_chart(px.bar(df.groupby('產品')['金額'].sum().sort_values(ascending=False).reset_index().head(12), x='金額', y='產品', orientation='h', color='金額', title="全球產品銷售排行榜"), use_container_width=True)

        # --- TAB 2: FOC ---
        with tabs[2]: 
            f_cols_list = ['培訓活動用針', '醫師酬勞', 'Workshop Training', 'Training/Rebate for JP', '市場贊助', '研究用針', '客訴補償', 'FOC樣品運輸']
            st.plotly_chart(px.bar(df[f_cols_list].sum().reset_index().rename(columns={'index':'類別', 0:'數量'}), x='類別', y='數量', color='類別', text_auto=True), use_container_width=True)
            target = st.selectbox("🔍 FOC 明細過濾：", options=["全部 FOC"] + f_cols_list)
            f_data_view = df[df['FOC總量']>0].copy()
            if target != "全部 FOC": f_data_view = f_data_view[f_data_view['FOC類別'] == target]
            st.dataframe(f_data_view[['銷貨日期', '國家', '客戶', '產品', '數量', '贈品量', '金額', '備註']], use_container_width=True)

        # --- TAB 3: 營運效率 (完整 5 大模組全數回歸！) ---
        with tabs[3]:
            st.subheader("⚡ 全球營運效率與模式深度解析")
            
            ce1, ce2 = st.columns(2)
            with ce1:
                st.markdown("""<div class="report-note">
                <b>1. 市場滲透度分析 (活躍客戶數)：</b><br>
                ● <b>直營診所模式：</b> 日本與台灣顯示實際開拓的診所總數量。<br>
                ● <b>經銷商模式：</b> 大陸包含「北京享贊」與「東莞双美」共 2 間；東南亞各國、德國與歐美各為 1 間經銷商。
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
                ● <b>意義：</b> 經銷商大宗進貨 (數值高)；直營診所小量多次採購 (數值低)。<br>
                ● <b>分析重點：</b> 經銷商採購量若過低，代表頻繁報關，行政物流成本過重。
                </div>""", unsafe_allow_html=True)
                order_sz = df[df['收費量']>0].groupby(['國家', '商務模式']).agg({'收費量':'sum', '銷貨日期':'count'}).reset_index()
                order_sz['規模'] = (order_sz['收費量'] / (order_sz['銷貨日期'] + 0.0001)).round(1)
                st.plotly_chart(px.bar(order_sz.sort_values('規模', ascending=False), x='國家', y='規模', color='商務模式', title="平均單次訂單規模", text_auto=True), use_container_width=True)

            st.divider()
            ce3, ce4 = st.columns([2, 1])
            with ce3:
                st.markdown("""<div class="report-note">
                <b>3. 行銷投資回報 (FOC 轉換效率)：</b><br>
                ● <b>意義：</b> 每投入 1 支贈針(FOC)，平均換回幾支收費訂單。<br>
                ● <b>分析重點：</b> 評估各國 Workshop、示範針的實質「轉單變現力」。
                </div>""", unsafe_allow_html=True)
                eff_df = df.groupby('國家').agg({'收費量':'sum', 'FOC總量':'sum'}).reset_index()
                eff_df['效率'] = (eff_df['收費量'] / (eff_df['FOC總量'] + 0.001)).round(1)
                st.plotly_chart(px.bar(eff_df.sort_values('效率', ascending=False), x='國家', y='效率', title="行銷槓桿比 (1支FOC換回幾張訂單)", color='效率', text_auto=True), use_container_width=True)

            with ce4:
                st.markdown("""<div class="report-note">
                <b>4. 全球商務模式佔比：</b><br>
                顯示全球直營診所模式與代理經銷模式的營收結構比例。
                </div>""", unsafe_allow_html=True)
                st.plotly_chart(px.pie(df, names='商務模式', hole=0.5, color_discrete_sequence=['#0984E3', '#00B894'], title="全球模式營收佔比"), use_container_width=True)

            st.markdown("""<div class="report-note">
            <b>5. 全球採購季節性熱力圖 (Heatmap)：</b><br>
            ● <b>意義：</b> 顏色越深代表該月份進貨金額越多。<br>
            ● <b>分析重點：</b> 追蹤各國大型展會或促銷後的補貨節奏，提早預備庫存與差旅。
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
