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
    /* 強制顯示橫向拉桿，防止表格被切斷 */
    .stDataFrame { overflow-x: auto !important; width: 100% !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 智慧數據處理引擎 (解鎖歷史與雙系統) ---
def smart_normalize(df):
    if df is None or df.empty: return pd.DataFrame()
    
    # 預抓 BX 備註 (索引 75)
    bx_note = df.iloc[:, 75].astype(str).replace('nan', '') if df.shape[1] >= 76 else None
    
    # 清理標題：去空格、換行、轉大寫
    df.columns = [str(c).strip().replace('\n', '').upper() for c in df.columns]
    
    # [最強匹配字典] 針對舊系統進行全面對接
    m = {
        'DATE': ['銷貨日期', '單據日期', '日期', 'DATE', '銷貨日期A', '單據日', '成交日期', '開單日期', '日期_1'],
        'YEAR_ONLY': ['年度', '年', 'YEAR', '年度_1', '年份', '西元年', 'YYYY', 'FISCALYEAR', '會計年度', '西元'],
        'COUNTRY': ['國家', '地區', 'COUNTRY', '區域', '廠別名稱', '地  區', '省份', '市場', '國別'],
        'CUSTOMER': ['客戶簡稱', '客戶', '客戶名稱', 'CUSTOMER', '送貨客戶全名', '經銷商', '代理商'],
        'PRODUCT': ['品名', '業務品名', '產品名稱', 'PRODUCT', '品號', '業務規格', '品  名', '項目名稱'],
        'QTY': ['銷貨數量', '數量', '計價數量', '總數量', 'QTY', '件數', '出貨數量', '數量_1'],
        'GIFT_QTY': ['贈/備品量', '贈品量', '贈品數量', 'GIFTQTY'],
        'PRICE': ['單價', '單價NT', 'PRICE', '單   價', '單  價', '單位成本'],
        'AMOUNT': ['本幣未稅金額', '本幣合計', '金額', '未稅金額', 'AMOUNT', '總計', '本幣金額', '銷貨金額', '幣別金額', '合計', '銷貨淨額', '未稅本幣']
    }

    def find_col(key):
        for name in m.get(key, []):
            if name.upper() in df.columns: return name.upper()
        return None

    p_df = pd.DataFrame()
    
    # 智慧時間識別：優先抓年度數字
    c_year = find_col('YEAR')
    c_date = find_col('DATE')
    
    if c_year:
        p_df['年度'] = pd.to_numeric(df[c_year], errors='coerce').fillna(0).astype(int)
        p_df['銷貨日期'] = pd.to_datetime(p_df['年度'].astype(str) + '-01-01', errors='coerce')
    elif c_date:
        dates = pd.to_datetime(df[c_date], errors='coerce')
        if dates.isna().all(): # 如果日期欄存的是 2012 這種數字
            p_df['年度'] = pd.to_numeric(df[c_date], errors='coerce').fillna(0).astype(int)
            p_df['銷貨日期'] = pd.to_datetime(p_df['年度'].astype(str) + '-01-01', errors='coerce')
        else:
            p_df['銷貨日期'] = dates
            p_df['年度'] = p_df['銷貨日期'].dt.year.fillna(0).astype(int)
    else:
        return pd.DataFrame()

    # 解鎖歷史：只要大於 2000 年都全收
    p_df = p_df[p_df['年度'] >= 2000].copy()
    if p_df.empty: return pd.DataFrame()

    p_df['月份'] = p_df['銷貨日期'].dt.month.fillna(1).astype(int)
    
    # 國家與區域分類
    c_country = find_col('COUNTRY')
    p_df['國家'] = df[c_country].loc[p_df.index].fillna('台灣').astype(str) if c_country else '台灣'
    def classify_region(c):
        c_str = str(c).strip()
        if any(x in c_str for x in ['台灣', '臺灣']): return '台灣市場', '經銷商模式'
        if any(x in c_str for x in ['大陸', '中國', 'CHINA', '大陸市場']): return '中國市場', '經銷商模式'
        if '日本' in c_str or 'JAPAN' in c_str.upper(): return '海外市場', '直營診所模式'
        return '海外市場', '經銷商模式'
    p_df['市場區域'], p_df['商務模式'] = zip(*p_df['國家'].apply(classify_region))

    # 官方名稱修正
    def get_official(name, country):
        n, c = str(name).upper(), str(country).upper()
        if 'VANGUARD' in n:
            if '新加坡' in country or 'SINGAPORE' in c: return 'VANGUARD AESTHETICS PTE. LTD.'
            if '菲律賓' in country or 'PHILIPPINES' in c: return 'VANGUARD AESTHETICS OPC'
            if '馬來西亞' in country or 'MALAYSIA' in c: return 'Vanguard Aesthetics Sdn Bhd'
        if 'QUALTECH' in n: return 'Qualtech Consulting (Thailand)'
        return re.sub(r'\s*(PTE\.?\s*LTD\.?|SDN\.?\s*BHD\.?|OPC|CORP\.?|INC\.?|CO\.?|LTD\.?)$', '', n).strip()
    
    c_cust = find_col('CUSTOMER')
    cust_data = df[c_cust].loc[p_df.index].astype(str) if c_cust else pd.Series(['未知客戶']*len(p_df))
    p_df['客戶'] = [get_official(n, c) for n, c in zip(cust_data, p_df['國家'])]

    # 數值轉換
    def to_num(key):
        col = find_col(key)
        return pd.to_numeric(df[col].loc[p_df.index].astype(str).str.replace(',', ''), errors='coerce').fillna(0.0) if col else pd.Series([0.0]*len(p_df)).loc[p_df.index]

    p_df['產品'] = df[find_col('PRODUCT')].loc[p_df.index].astype(str).str.replace('Sunmax DeusaDerm', 'Sunmax Deusaderm VITAL', case=False).str.replace('VITAL VITAL', 'VITAL', case=False) if find_col('PRODUCT') else "未知產品"
    p_df['數量'] = to_num('QTY')
    p_df['贈品量'] = to_num('GIFT_QTY')
    p_df['單價'] = to_num('PRICE')
    p_df['金額'] = to_num('AMOUNT')
    p_df['備註'] = bx_note.loc[p_df.index].values if bx_note is not None else (df[find_col('NOTE')].loc[p_df.index].astype(str).replace('nan', '') if find_col('NOTE') else "")

    # FOC 邏輯
    p_df['實際FOC數量'] = np.where((p_df['單價'] <= 0) | (p_df['金額'] <= 0), p_df['數量'], 0.0) + p_df['贈品量']
    foc_cats = ['培訓活動用針', '醫師酬勞', 'Workshop Training', 'Training/Rebate for JP', '市場贊助', '研究用針', '客訴補償', 'FOC樣品運輸']
    for cat in foc_cats: p_df[cat] = 0.0
    def classify_foc_row(row):
        f_qty = row['實際FOC數量']
        if f_qty <= 0: return '非FOC'
        rem = str(row['備註']).lower()
        if 'workshop training' in rem: return 'Workshop Training'
        if any(x in rem for x in ['實操針', '示範針', '會員實操針', '培訓針', '培訓施打用針', 'demo', '培訓活動施打用針']): return '培訓活動用針'
        if any(x in rem for x in ['酬勞針', '講師針', 'speech']): return '醫師酬勞'
        if any(x in rem for x in ['rebate', 'training']): return 'Training/Rebate for JP'
        if any(x in rem for x in ['sponsorship', 'launch event', 'launch']): return '市場贊助'
        if any(x in rem for x in ['sample needles', 'irb']): return '研究用針'
        if any(x in rem for x in ['complaints', '客訴', '補償', '瑕疵', '更換', 'quality', 'customer complaints']): return '客訴補償'
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
        all_dfs = []
        for name, df_sheet in sheets.items():
            norm = smart_normalize(df_sheet)
            if not norm.empty: all_dfs.append(norm)
        return pd.concat(all_dfs, ignore_index=True) if all_dfs else None
    except Exception as e:
        st.error(f"數據讀取失敗: {e}")
        return None

# --- 4. 主介面 ---
with st.sidebar:
    st.title("⚙️ 數據管理")
    admin_mode = st.toggle("管理員模式")
    if admin_mode:
        pwd = st.text_input("管理密碼", type="password")
        if pwd == "sunmax888":
            st.file_uploader("上傳 Excel (臨時)", type=["xlsx"])
            if st.button("更新數據快取"): st.cache_data.clear()

st.title("📊 双美全球銷售數據")
df_all = load_all_data()

if df_all is not None:
    # 頂部篩選器
    sc1, sc2, sc3 = st.columns([1, 1, 2])
    sel_area = sc1.multiselect("區域", ['台灣市場', '中國市場', '海外市場'], default=['海外市場', '中國市場'])
    available_countries = sorted(df_all[df_all['市場區域'].isin(sel_area)]['國家'].unique())
    sel_countries = sc3.multiselect("國家", available_countries, default=available_countries)
    
    # --- 關鍵修正：App 打開時預設「全選」所有年份，找回 2022 之前的數據 ---
    all_available_yrs = sorted([int(y) for y in df_all['年度'].unique() if y > 0])
    sel_yrs = sc2.multiselect("年度", options=all_available_yrs, default=all_available_yrs)

    df = df_all[(df_all['年度'].isin(sel_yrs)) & (df_all['國家'].isin(sel_countries)) & (df_all['市場區域'].isin(sel_area))]

    if not df.empty:
        yr_label = f"{min(sel_yrs)}年-{max(sel_yrs)}年" if len(sel_yrs)>1 else f"{sel_yrs[0]}年"
        st.write(f"🔍 **數據統計區間：{yr_label}**")
        
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("總銷售金額", f"NT${df['金額'].sum():,.0f}")
        k2.metric("收費訂單量", f"{df['收費量'].sum():,.0f}")
        k3.metric("FOC 贈送總量", f"{df['FOC總量'].sum():,.0f}")
        total_q = df['收費量'].sum() + df['FOC總量'].sum()
        k4.metric("平均贈針比", f"{(df['FOC總量'].sum() / (total_q + 0.0001) * 100):.1%}")
        k5.metric("活躍國家數", f"{df['國家'].nunique()}")

        tabs = st.tabs(["📊 市場佔比與YoY", "🎯 產品排行", "📉 FOC 專項分析", "⚡ 營運效率(進階)", "🔍 明細查詢"])
        
        with tabs[0]: 
            metric_opt = st.selectbox("分析指標", ["金額", "收費量", "FOC總量"])
            m_col = {'金額': '金額', '收費量': '收費量', 'FOC總量': 'FOC總量'}[metric_opt]
            st.plotly_chart(px.pie(df.groupby('國家')[m_col].sum().reset_index(), values=m_col, names='國家', hole=0.4, title="各國份額佔比", color_discrete_sequence=px.colors.qualitative.Pastel), use_container_width=True)
            
            st.divider()
            st.write("#### 📈 逐年成長趨勢表 (YoY%) - 若欄位過多請左右滑動")
            # 建立透視表
            pivot = df.pivot_table(index='國家', columns='年度', values=m_col, aggfunc='sum').fillna(0)
            sorted_y = sorted(pivot.columns); disp = pivot.copy()
            for i in range(1, len(sorted_y)):
                curr, prev = sorted_y[i], sorted_y[i-1]
                disp[f"{curr}年 成長%"] = np.where(pivot[prev]==0, np.nan, (pivot[curr]-pivot[prev])/pivot[prev]*100)
            
            final_f_cols = []
            f_dict = {}
            for y in sorted_y:
                y_c = f"{int(y)}年"; disp = disp.rename(columns={y: y_c})
                final_f_cols.append(y_c); f_dict[y_c] = "{:,.0f}"
                g_c = f"{int(y)}年 成長%"
                if g_c in disp.columns: final_f_cols.append(g_c); f_dict[g_c] = "{:.1f}%"
            
            def color_logic(v, n):
                if '成長%' in str(n) and pd.notna(v):
                    return 'color: #D32F2F; font-weight: bold' if v > 0.001 else 'color: #388E3C; font-weight: bold'
                return ''
            # 水平拉桿啟用 + 固定高度 450
            st.dataframe(disp[final_f_cols].style.format(f_dict, na_rep="-").apply(lambda x: [color_logic(v, x.name) for v in x], axis=0), height=450, use_container_width=True)

        with tabs[3]: # 營運效率 (藍色說明永久固定)
            st.subheader("⚡ 全球營運效率與模式深度解析")
            ce1, ce2 = st.columns(2)
            with ce1:
                st.markdown('<div class="report-note"><b>1. 市場滲透度分析 (活躍客戶數)：</b><br>● 日本(直營)顯示診所總量。經銷商國家應為 1。</div>', unsafe_allow_html=True)
                active_c = df.groupby(['國家', '商務模式'])['客戶'].nunique().reset_index().sort_values('客戶', ascending=False)
                st.plotly_chart(px.bar(active_c, x='國家', y='客戶', color='商務模式', title="活躍客戶/診所總數", text_auto=True), use_container_width=True)
                with st.expander("📋 查看各國公司名單"):
                    cust_dt = df.groupby(['國家', '商務模式'])['客戶'].unique().reset_index()
                    cust_dt['清單'] = cust_dt['客戶'].apply(lambda x: "\n".join([f"• {name}" for name in x]))
                    st.write(cust_dt[['國家', '商務模式', '清單']].to_html(escape=False).replace('\\n', '<br>'), unsafe_allow_html=True)
            with ce2:
                st.markdown('<div class="report-note"><b>2. 物流模式分析 (平均單次規模)：</b><br>● 數值越高代表單次採購量大，物流效率越好。</div>', unsafe_allow_html=True)
                order_sz = df[df['收費量']>0].groupby(['國家', '商務模式']).agg({'收費量':'sum', '銷貨日期':'count'}).reset_index()
                order_sz['規模'] = (order_sz['收費量'] / (order_sz['銷貨日期'] + 0.0001)).round(1)
                st.plotly_chart(px.bar(order_sz.sort_values('規模', ascending=False), x='國家', y='規模', color='商務模式', title="平均單次訂單規模", text_auto=True), use_container_width=True)
            st.divider()
            ce3, ce4 = st.columns([2, 1])
            with ce3:
                st.markdown('<div class="report-note"><b>3. 行銷投資回報 (FOC 轉換效率)：</b><br>● 每投入1支贈針換回幾張訂單。數值越高代表投資報酬率越高。</div>', unsafe_allow_html=True)
                eff_df = df.groupby('國家').agg({'收費量':'sum', 'FOC總量':'sum'}).reset_index()
                eff_df['效率'] = (eff_df['收費量'] / (eff_df['FOC總量'] + 0.001)).round(1)
                st.plotly_chart(px.bar(eff_df.sort_values('效率', ascending=False), x='國家', y='效率', title="行銷槓桿比", color='效率', text_auto=True), use_container_width=True)
            with ce4:
                st.markdown('<div class="report-note"><b>4. 營收模式佔比：</b><br>顯示全球直營與經銷的營收結構。</div>', unsafe_allow_html=True)
                st.plotly_chart(px.pie(df, names='商務模式', hole=0.5, color_discrete_sequence=['#0984E3', '#00B894'], title="全球模式佔比"), use_container_width=True)
            
            heat_df = df.groupby(['國家', '月份'])['金額'].sum().reset_index()
            fig_h = px.density_heatmap(heat_df, x='月份', y='國家', z='金額', color_continuous_scale='Blues', nbinsx=12, range_x=[0.5, 12.5], text_auto='.2s')
            fig_h.update_xaxes(tickmode='linear', tick0=1, dtick=1, ticktext=[f"{i}月" for i in range(1,13)], tickvals=list(range(1,13)))
            st.plotly_chart(fig_h, use_container_width=True)

        with tabs[1]: st.plotly_chart(px.bar(df.groupby('產品')['金額'].sum().sort_values(ascending=False).reset_index().head(12), x='金額', y='產品', orientation='h', color='金額', title="全球產品銷售排行榜"), use_container_width=True)
        with tabs[2]: # FOC
            f_cols = ['培訓活動用針', '醫師酬勞', 'Workshop Training', 'Training/Rebate for JP', '市場贊助', '研究用針', '客訴補償', 'FOC樣品運輸']
            st.plotly_chart(px.bar(df[f_cols].sum().reset_index().rename(columns={'index':'類別', 0:'數量'}), x='類別', y='數量', color='類別', text_auto=True), use_container_width=True)
            target = st.selectbox("🔍 FOC 明細過濾：", options=["全部 FOC"] + f_cols)
            f_v = df[df['FOC總量']>0].copy()
            if target != "全部 FOC": f_v = f_v[foc_v['FOC類別'] == target]
            st.dataframe(foc_v[['銷貨日期', '國家', '客戶', '產品', '數量', '贈品量', '金額', '備註']], use_container_width=True)
        with tabs[4]: # 明細
            q = st.text_input("搜尋關鍵字...").lower()
            q_df = df.copy()
            if q: q_df = q_df[q_df['客戶'].str.lower().str.contains(q, na=False) | q_df['產品'].str.lower().str.contains(q, na=False)]
            st.dataframe(q_df[['銷貨日期', '市場區域', '商務模式', '國家', '客戶', '產品', '數量', '金額', '備註']], use_container_width=True)
else:
    st.info("👋 管理員您好，請確保 `data.xlsx` 已上傳至 GitHub 根目錄。")
