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

# --- AUTO-ADAPTIVE UI & GLASSMORPHISM CSS ---
# Using CSS variables so it looks gorgeous in BOTH Light and Dark modes
st.markdown("""
<style>
    /* Top Gradient Header - Works on Light & Dark */
    h1 {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text; 
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        text-align: center;
        padding-bottom: 10px;
    }

    /* Subtitle styling */
    .subtitle-text {
        text-align: center; 
        color: var(--text-color); 
        opacity: 0.7;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* KPI Metric Cards (Universal Glassmorphic) */
    div[data-testid="metric-container"] {
        background: rgba(130, 140, 150, 0.1); 
        border: 1px solid rgba(130, 140, 150, 0.2);
        backdrop-filter: blur(10px); 
        -webkit-backdrop-filter: blur(10px);
        border-radius: 12px; 
        padding: 20px; 
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease-in-out;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px); 
        border: 1px solid var(--primary-color);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }
    
    /* KPI Value - Uses native text color */
    div[data-testid="stMetricValue"] { 
        color: var(--text-color); 
        font-weight: 800; 
        font-size: 2.2rem;
    }

    /* Tabs Styling - Adaptive */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(130, 140, 150, 0.05);
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        color: var(--text-color);
        opacity: 0.7;
        border: 1px solid transparent;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(130, 140, 150, 0.15);
        opacity: 1;
        border: 1px solid rgba(130, 140, 150, 0.2);
        border-bottom: 3px solid var(--primary-color);
    }

    /* AI Custom Container */
    .ai-container {
        background: rgba(130, 140, 150, 0.08); 
        border: 1px solid rgba(130, 140, 150, 0.2); 
        border-radius: 12px; 
        padding: 25px; 
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛒 Universal Sales Intelligence Platform")
st.markdown("<div class='subtitle-text'>Upload ANY sales dataset. The system will automatically generate KPIs, graphs, and AI insights.</div>", unsafe_allow_html=True)
st.markdown("---")

# --- SIDEBAR: UNIVERSAL FILE UPLOAD ---
st.sidebar.markdown("### 📂 1. File Upload")
uploaded_file = st.sidebar.file_uploader("Drag and drop CSV here", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the uploaded CSV
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ Dataset loaded successfully!")
        
        # Auto-detect column types to prevent errors
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()

        if not numeric_cols:
            st.sidebar.error("⚠️ No numeric columns found for calculations.")
        else:
            st.sidebar.markdown("### 🔗 2. User Column Mapping")
            
            # Select Numeric columns for math
            revenue_col = st.sidebar.selectbox("Sales/Revenue Column", numeric_cols, index=0)
            qty_col = st.sidebar.selectbox("Quantity Column", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
            
            # Select Categorical column for grouping
            category_col = st.sidebar.selectbox("Category Column", categorical_cols if categorical_cols else df.columns, index=0)

            # --- DATA CLEANING (Before calculating KPIs) ---
            is_messy = df.isnull().values.any() or df.duplicated().any()
            
            if is_messy:
                cleaned_df = df.copy()
                
                # Impute missing numericals with Median
                for col in cleaned_df.select_dtypes(include=['number']).columns:
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
                    
                # Impute missing categoricals with Mode
                for col in cleaned_df.select_dtypes(include=['object']).columns:
                    if not cleaned_df[col].mode().empty:
                        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])
                        
                # Remove duplicates
                cleaned_df = cleaned_df.drop_duplicates()
                
                active_df = cleaned_df
            else:
                active_df = df

            # --- CALCULATE UNIVERSAL KPIs (Using Cleaned Data) ---
            total_revenue = active_df[revenue_col].sum()
            total_orders = len(active_df)
            total_units = active_df[qty_col].sum()

            # --- TABS UI ---
            tab1, tab2, tab3 = st.tabs(["📊 Visual Dashboard", "🗂️ Data Preview", "🤖 GenAI Assistant"])

            # TAB 1: VISUAL DASHBOARD
            with tab1:
                # KPI Cards
                col1, col2, col3 = st.columns(3)
                with col1: st.metric(f"Total Revenue", f"${total_revenue:,.2f}")
                with col2: st.metric("Total Orders", f"{total_orders:,}")
                with col3: st.metric(f"Total Units", f"{total_units:,.0f}")
                
                st.markdown("<br><br>", unsafe_allow_html=True)

                # Dynamic Graphs
                col_chart1, col_chart2 = st.columns([1.5, 1])
                
                # Using Streamlit's native theme engine for Plotly (Adapts to Light/Dark automatically)
                with col_chart1:
                    cat_revenue = active_df.groupby(category_col)[revenue_col].sum().reset_index().sort_values(by=revenue_col, ascending=False).head(10)
                    fig_bar = px.bar(cat_revenue, x=revenue_col, y=category_col, orientation='h', 
                                     title=f"Top Performing Categories", 
                                     color=revenue_col, color_continuous_scale="Tealgrn")
                    fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=20, r=20, t=50, b=20))
                    # theme="streamlit" ensures it looks perfect in both light and dark mode
                    st.plotly_chart(fig_bar, use_container_width=True, theme="streamlit")
                    
                with col_chart2:
                    cat_qty = active_df.groupby(category_col)[qty_col].sum().reset_index().sort_values(by=qty_col, ascending=False).head(10)
                    fig_pie = px.pie(cat_qty, values=qty_col, names=category_col, hole=0.5, 
                                     title=f"Category Distribution")
                    fig_pie.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5), margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig_pie, use_container_width=True, theme="streamlit")

            # TAB 2: DATA PREVIEW
            with tab2:
                if is_messy:
                    st.warning("🚨 Messy Data Detected")
                    st.info("🟢 **Auto-Cleaning Active:** The system caught missing data and automatically imputed values to prevent calculation errors.")
                    
                    sub_tab1, sub_tab2 = st.tabs(["🔴 Before (Raw Data)", "🟢 After (Cleaned Data)"])
                    
                    with sub_tab1:
                        st.dataframe(df.head(10), use_container_width=True) 
                        st.caption(f"Found: {df.isnull().sum().sum()} missing values & {df.duplicated().sum()} duplicates.")
                        
                    with sub_tab2:
                        st.dataframe(active_df.head(10), use_container_width=True)
                        st.caption("Missing values replaced with Median (Numeric) and Mode (Categorical). Duplicates removed.")
                else:
                    st.success("✅ Clean Dataset")
                    st.dataframe(df.head(10), use_container_width=True)

            # TAB 3: GENAI ASSISTANT
            with tab3:
                st.markdown("""
                    <div class="ai-container">
                        <h3 style="margin-top: 0;">💬 Ask AI About This Dataset</h3>
                        <p style="opacity: 0.8;">Generate insights, summaries, and business emails directly from your data.</p>
                    </div>
                """, unsafe_allow_html=True)
                
                api_key = os.getenv("GEMINI_API_KEY")
                
                if not api_key:
                    st.error("⚠️ API Key not found. Please check your `.env` file.")
                else:
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
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        user_question = st.chat_input("Ask GenAI (e.g., Draft a summary email for the team...)")
                        
                        if user_question:
                            with st.chat_message("user"):
                                st.write(user_question)
                                
                            with st.spinner("AI is analyzing the dataset..."):
                                prompt = f"{business_context}\n\nUser Question: {user_question}"
                                response = model.generate_content(prompt)
                                
                            with st.chat_message("assistant"):
                                st.write(response.text)
                    except Exception as e:
                        st.error(f"GenAI Error: {e}")

    except Exception as read_error:
        st.error(f"Could not read the file. Please ensure it's a valid CSV. Error: {read_error}")
else:
    # Initial Screen
    st.info("👈 Drop a CSV file in the sidebar to initiate the dashboard.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 1️⃣ Upload")
        st.caption("Drag and drop any tabular sales data.")
    with col2:
        st.markdown("#### 2️⃣ Map")
        st.caption("Dynamic mapping identifies your structure.")
    with col3:
        st.markdown("#### 3️⃣ Analyze")
        st.caption("Interact with AI-driven charts and insights.")