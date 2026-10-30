import sys
import asyncio

# Fix for Windows asyncio bug
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load Environment Variables from .env file
load_dotenv()

st.set_page_config(page_title="Retail Analytics Dashboard", page_icon="📊", layout="wide")

# --- CUSTOM UI & GLASSMORPHISM CSS ---
st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background: rgba(30, 45, 65, 0.4); border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
        border-radius: 15px; padding: 20px; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease-in-out;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px); border: 1px solid rgba(77, 184, 255, 0.5);
        box-shadow: 0 12px 40px 0 rgba(77, 184, 255, 0.2);
    }
    div[data-testid="stMetricValue"] { color: #4DB8FF; font-weight: 700; }
    h1 {
        background: -webkit-linear-gradient(45deg, #4DB8FF, #00FF88);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
</style>
""", unsafe_allow_html=True)

DB_PATH = os.path.join("sql", "retail_warehouse.db")

@st.cache_data
def load_data(query):
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Database Connection Error: {e}")
        return pd.DataFrame()

st.title("🛒 Retail Sales Intelligence")
st.markdown("Connects directly to the Star Schema SQLite Data Warehouse with AI/ML Customer Segmentation.")

# --- SIDEBAR FILTERS ---
st.sidebar.markdown("### 🎛️ Dashboard Filters")
country_list_query = "SELECT DISTINCT Country FROM dim_customer ORDER BY Country"
countries_df = load_data(country_list_query)
selected_country = st.sidebar.selectbox("Select Country", ["All"] + countries_df['Country'].tolist())

where_clause = "WHERE f.IsCancelled = 0"
if selected_country != "All":
    where_clause += f" AND c.Country = '{selected_country}'"

# Fetch Global Data context for GenAI
kpi_query = f"SELECT COUNT(DISTINCT f.InvoiceNo) as TotalOrders, SUM(f.SalesAmount) as TotalRevenue, SUM(f.Quantity) as TotalUnitsSold FROM fact_sales f JOIN dim_customer c ON f.CustomerID = c.CustomerID {where_clause}"
kpis = load_data(kpi_query)

# --- TABS FOR CLEAN UI ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Sales Overview", "👥 AI Customer Segments", "🛍️ Product Recommendations", "🤖 GenAI Assistant"])

# TAB 1: SALES OVERVIEW
with tab1:
    if not kpis.empty:
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Total Revenue", f"£{kpis['TotalRevenue'].iloc[0]:,.2f}")
        with col2: st.metric("Total Orders", f"{kpis['TotalOrders'].iloc[0]:,}")
        with col3: st.metric("Units Sold", f"{kpis['TotalUnitsSold'].iloc[0]:,}")
    
    st.markdown("<br>", unsafe_allow_html=True)

    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        trend_query = f"SELECT d.Year, d.Month, SUM(f.SalesAmount) as Revenue FROM fact_sales f JOIN dim_date d ON f.DateKey = d.DateKey JOIN dim_customer c ON f.CustomerID = c.CustomerID {where_clause} GROUP BY d.Year, d.Month ORDER BY d.Year, d.Month"
        trend_data = load_data(trend_query)
        if not trend_data.empty:
            trend_data['MonthYear'] = trend_data['Year'].astype(str) + "-" + trend_data['Month'].astype(str).str.zfill(2)
            fig_line = px.line(trend_data, x='MonthYear', y='Revenue', markers=True, title="📈 Revenue Over Time", template="plotly_dark")
            fig_line.update_traces(line=dict(color="#00FF88", width=3), marker=dict(size=8, color="#4DB8FF"))
            fig_line.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_line, width="stretch")

    with col_chart2:
        country_query = f"SELECT c.Country, SUM(f.SalesAmount) as Revenue FROM fact_sales f JOIN dim_customer c ON f.CustomerID = c.CustomerID {where_clause} GROUP BY c.Country ORDER BY Revenue DESC LIMIT 10"
        country_sales = load_data(country_query)
        if not country_sales.empty:
            fig_bar = px.bar(country_sales, x="Revenue", y="Country", orientation='h', text_auto='.2s', color="Revenue", color_continuous_scale="Tealgrn", title="🌍 Top 10 Countries by Revenue", template="plotly_dark")
            fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_bar, width="stretch")

# TAB 2: AI CUSTOMER SEGMENTS
with tab2:
    st.markdown("#### 🤖 RFM + K-Means Clustering")
    seg_query = "SELECT * FROM customer_segments"
    segments_df = load_data(seg_query)
    if not segments_df.empty:
        col_seg1, col_seg2 = st.columns(2)
        with col_seg1:
            seg_counts = segments_df['SegmentLabel'].value_counts().reset_index()
            seg_counts.columns = ['Segment', 'Count']
            fig_pie = px.pie(seg_counts, values='Count', names='Segment', hole=0.5, title="Customer Distribution", template="plotly_dark", color_discrete_sequence=['#4DB8FF', '#00FF88', '#FF6B6B', '#FCA311'])
            fig_pie.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pie, width="stretch")
        with col_seg2:
            fig_scatter = px.scatter(segments_df, x='Recency', y='Monetary', color='SegmentLabel', title="Recency vs Monetary Value", template="plotly_dark", color_discrete_sequence=['#4DB8FF', '#00FF88', '#FF6B6B', '#FCA311'])
            fig_scatter.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_scatter, width="stretch")

# TAB 3: PRODUCT RECOMMENDATIONS
with tab3:
    st.markdown("#### 🛍️ Market Basket Analysis (Apriori Algorithm)")
    rules_query = "SELECT antecedents AS 'If Customer Buys...', consequents AS 'They Also Buy...', confidence AS 'Confidence (%)', lift AS 'Lift (Strength)' FROM market_basket_rules ORDER BY lift DESC LIMIT 20"
    rules_df = load_data(rules_query)
    if not rules_df.empty:
        rules_df['Confidence (%)'] = (rules_df['Confidence (%)'] * 100).round(2).astype(str) + '%'
        rules_df['Lift (Strength)'] = rules_df['Lift (Strength)'].round(2)
        st.dataframe(rules_df, use_container_width=True, hide_index=True)

# TAB 4: GENAI ASSISTANT
with tab4:
    st.subheader("💬 Ask the Data (Powered by Gemini)")
    st.markdown("Ask questions about revenue, top countries, or customer segments based on the current dashboard filters.")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.warning("⚠️ API Key not found. Please create a `.env` file and add your `GEMINI_API_KEY`.")
    else:
        # Construct business context to feed the AI
        business_context = f"""
        You are a Retail Analytics Assistant. Answer the user's question based strictly on this current data context:
        - Selected Filter: {selected_country}
        - Total Revenue: £{kpis['TotalRevenue'].iloc[0]:,.2f} if data exists.
        - Total Orders: {kpis['TotalOrders'].iloc[0]:,} if data exists.
        - Total Units Sold: {kpis['TotalUnitsSold'].iloc[0]:,} if data exists.
        Do not make up numbers. Keep your answers concise, professional, and directly address the business value.
        """
        
        genai.configure(api_key=api_key)
        
        try:
            # Using the exact model recommended by the Google API
            model = genai.GenerativeModel('gemini-3.8-flash')
            
            user_question = st.text_input("Type your question here (e.g., 'What is the total revenue?'):")
            
            if st.button("Ask Assistant") and user_question:
                with st.spinner("Analyzing data..."):
                    try:
                        prompt = f"{business_context}\n\nUser Question: {user_question}"
                        response = model.generate_content(prompt)
                        st.success("🤖 Assistant Response:")
                        st.write(response.text)
                    except Exception as e:
                        st.error(f"Error generating response: {e}")
        except Exception as api_error:
            st.error(f"API Configuration Error: {api_error}. Please verify your API key in the .env file.")

st.markdown("<hr style='border:1px solid rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
st.caption("Data source: UCI Online Retail Dataset | UI: Glassmorphism | Architecture: Star Schema SQLite")