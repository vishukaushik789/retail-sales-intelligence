import sys
import asyncio

# Fix for Windows asyncio bug
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
import pandas as pd
import plotly.express as px
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

st.set_page_config(page_title="Universal Data Analytics", page_icon="📊", layout="wide")

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
    }
    div[data-testid="stMetricValue"] { color: #4DB8FF; font-weight: 700; }
    h1 {
        background: -webkit-linear-gradient(45deg, #4DB8FF, #00FF88);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛒 Universal Sales Intelligence Platform")
st.markdown("Upload **ANY** sales dataset. The system will automatically generate KPIs, graphs, and AI insights.")

# --- SIDEBAR: UNIVERSAL FILE UPLOAD ---
st.sidebar.markdown("### 📂 1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the uploaded CSV
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("Data loaded successfully!")
        
        # Auto-detect column types to prevent errors
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()

        if not numeric_cols:
            st.error("⚠️ Your dataset must have at least one numeric column (like Price or Quantity) for calculations.")
        else:
            st.sidebar.markdown("### 🔗 2. Map Your Columns")
            st.sidebar.caption("Tell the system which columns to analyze.")
            
            # Select Numeric columns for math
            revenue_col = st.sidebar.selectbox("Select Sales/Revenue Column", numeric_cols, index=0)
            qty_col = st.sidebar.selectbox("Select Quantity Column", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
            
            # Select Categorical column for grouping
            category_col = st.sidebar.selectbox("Select Graph Category (e.g., Region, Product)", categorical_cols if categorical_cols else df.columns, index=0)

            # --- CALCULATE UNIVERSAL KPIs ---
            total_revenue = df[revenue_col].sum()
            total_orders = len(df)
            total_units = df[qty_col].sum()

            # --- TABS UI ---
            tab1, tab2, tab3 = st.tabs(["📊 Visual Dashboard", "🗂️ Data Preview", "🤖 GenAI Assistant"])

            # TAB 1: VISUAL DASHBOARD
            with tab1:
                # KPI Cards
                col1, col2, col3 = st.columns(3)
                with col1: st.metric(f"Total {revenue_col}", f"₹ {total_revenue:,.2f}")
                with col2: st.metric("Total Records", f"{total_orders:,}")
                with col3: st.metric(f"Total {qty_col}", f"{total_units:,.0f}")
                
                st.markdown("<br>", unsafe_allow_html=True)

                # Dynamic Graphs
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    # Bar Chart
                    cat_revenue = df.groupby(category_col)[revenue_col].sum().reset_index().sort_values(by=revenue_col, ascending=False).head(10)
                    fig_bar = px.bar(cat_revenue, x=revenue_col, y=category_col, orientation='h', text_auto='.2s', color=revenue_col, color_continuous_scale="Tealgrn", title=f"🏆 Top 10 {category_col} by {revenue_col}", template="plotly_dark")
                    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_bar, width="stretch")
                    
                with col_chart2:
                    # Pie Chart
                    cat_qty = df.groupby(category_col)[qty_col].sum().reset_index().sort_values(by=qty_col, ascending=False).head(10)
                    fig_pie = px.pie(cat_qty, values=qty_col, names=category_col, hole=0.5, title=f"📊 Share of {qty_col} by {category_col}", template="plotly_dark", color_discrete_sequence=['#4DB8FF', '#00FF88', '#FF6B6B', '#FCA311', '#B19CD9'])
                    fig_pie.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_pie, width="stretch")

            # TAB 2: DATA PREVIEW
            with tab2:
                st.markdown(f"#### Raw Data Preview ({total_orders} Rows)")
                st.dataframe(df.head(100), use_container_width=True)

            # TAB 3: GENAI ASSISTANT
            with tab3:
                st.subheader("💬 Ask AI About This Dataset")
                api_key = os.getenv("GEMINI_API_KEY")
                
                if not api_key:
                    st.warning("⚠️ API Key not found. Please check your `.env` file.")
                else:
                    # Create dynamic context
                    business_context = f"""
                    You are a Universal Data Analyst Assistant. Answer questions based ONLY on this current context:
                    - Dataset contains {total_orders} rows.
                    - Total {revenue_col}: {total_revenue:,.2f}
                    - Total {qty_col}: {total_units:,.0f}
                    - The main category being analyzed is: {category_col}
                    Keep answers concise and professional.
                    """
                    genai.configure(api_key=api_key)
                    
                    try:
                        model = genai.GenerativeModel('gemini-3.8-flash')
                        user_question = st.text_input("Type your question here (e.g., 'What is the total value?'):")
                        
                        if st.button("Ask AI") and user_question:
                            with st.spinner("Analyzing your custom data..."):
                                prompt = f"{business_context}\n\nUser Question: {user_question}"
                                response = model.generate_content(prompt)
                                st.success("🤖 AI Response:")
                                st.write(response.text)
                    except Exception as e:
                        st.error(f"GenAI Error: {e}")

    except Exception as read_error:
        st.error(f"Could not read the file. Please ensure it's a valid CSV. Error: {read_error}")
else:
    # Initial Screen when no file is uploaded
    st.info("👈 Please upload a CSV file from the sidebar to magically generate your dashboard!")
    st.markdown("### How it works:")
    st.markdown("- **1. Upload:** Drop any tabular CSV sales/business data.")
    st.markdown("- **2. Auto-Detect:** The system finds your numerical and categorical columns automatically.")
    st.markdown("- **3. AI Insights:** Get instant visualizations and interact with your data using Google Gemini AI.")