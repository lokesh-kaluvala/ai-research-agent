"""
AI Research Agent - Professional Streamlit Interface
Built with LangGraph + OpenAI + Tavily
"""

import streamlit as st
import os
from datetime import datetime
from typing import List, Dict
import time

# Core dependencies
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "AI-powered research agent using LangGraph & GPT-4"
    }
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --success-color: #10b981;
        --warning-color: #f59e0b;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    
    /* Card styling */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #6366f1;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin: 1rem 0;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.875rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    .status-researching {
        background: #dbeafe;
        color: #1e40af;
    }
    
    .status-analyzing {
        background: #fef3c7;
        color: #92400e;
    }
    
    .status-complete {
        background: #d1fae5;
        color: #065f46;
    }
    
    /* Source links */
    .source-link {
        background: #f3f4f6;
        padding: 0.75rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        border-left: 3px solid #6366f1;
    }
    
    .source-link a {
        color: #6366f1;
        text-decoration: none;
        font-weight: 500;
    }
    
    .source-link a:hover {
        text-decoration: underline;
    }
    
    /* Metrics */
    .metric-container {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #0c4a6e;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #475569;
        margin-top: 0.25rem;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }
    
    /* Animation for processing */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .processing {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 4px 6px rgba(99, 102, 241, 0.25);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(99, 102, 241, 0.35);
    }
</style>
""", unsafe_allow_html=True)

# ------- CONFIGURATION -------
class Config:
    """App configuration"""
    MODEL = "gpt-4o-mini"
    TEMPERATURE = 0.3
    MAX_REVISIONS = 3
    SEARCH_RESULTS = 10

# ------- STATE MANAGEMENT -------
def init_session_state():
    """Initialize Streamlit session state"""
    if 'research_history' not in st.session_state:
        st.session_state.research_history = []
    if 'current_research' not in st.session_state:
        st.session_state.current_research = None
    if 'api_keys_set' not in st.session_state:
        st.session_state.api_keys_set = False

# ------- RESEARCH AGENT (Simplified for Streamlit) -------
class StreamlitResearchAgent:
    """Research agent optimized for Streamlit"""
    
    def __init__(self, openai_key: str, tavily_key: str):
        self.llm = ChatOpenAI(
            model=Config.MODEL,
            temperature=Config.TEMPERATURE,
            api_key=openai_key,
            request_timeout=30
        )
        self.search_tool = TavilySearchResults(
            max_results=Config.SEARCH_RESULTS,
            api_key=tavily_key,
            search_depth="advanced"
        )
    
    def research(self, query: str, progress_callback=None) -> Dict:
        """Execute research with progress updates"""
        results = {
            'query': query,
            'iterations': [],
            'final_analysis': '',
            'sources': [],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_sources': 0
        }
        
        all_data = []
        all_urls = []
        
        for iteration in range(Config.MAX_REVISIONS):
            if progress_callback:
                progress_callback(f"🔍 Research iteration {iteration + 1}/{Config.MAX_REVISIONS}")
            
            # Refine query based on iteration
            if iteration == 0:
                refined = f"{query} latest news 2026 market analysis"
            elif iteration == 1:
                refined = f"{query} expert analysis forecasts detailed"
            else:
                refined = f"{query} comprehensive report statistics"
            
            # Search
            try:
                search_results = self.search_tool.invoke(refined)
                
                iteration_data = []
                iteration_urls = []
                
                for r in search_results:
                    content = r.get('content', '').strip()
                    url = r.get('url', '')
                    
                    if len(content) > 50:
                        iteration_data.append(content)
                        iteration_urls.append(url)
                        all_data.append(content)
                        if url not in all_urls:
                            all_urls.append(url)
                
                results['iterations'].append({
                    'iteration': iteration + 1,
                    'query': refined,
                    'sources_found': len(iteration_data)
                })
                
            except Exception as e:
                results['iterations'].append({
                    'iteration': iteration + 1,
                    'query': refined,
                    'error': str(e)
                })
                continue
        
        # Final analysis
        if progress_callback:
            progress_callback("🧠 Analyzing all gathered data...")
        
        combined_data = "\n\n---\n\n".join(all_data[:15])  # Limit to prevent token overflow
        
        analysis_prompt = f"""
Analyze this research on: {query}

DATA FROM {len(all_data)} SOURCES:
{combined_data[:6000]}

Provide a comprehensive analysis with:

## 🎯 Executive Summary
[2-3 sentence overview]

## 🏢 Key Market Players
[List 3-5 major companies/entities with brief context]

## 📈 Bull Case for 2026
[3-4 positive factors/opportunities]

## 📉 Bear Case for 2026
[3-4 risks/challenges]

## 💡 Key Insights
[3-5 critical takeaways]

## 🎲 Confidence Assessment
[HIGH/MEDIUM/LOW with justification]

Keep it professional, data-driven, and actionable.
"""
        
        try:
            response = self.llm.invoke([
                SystemMessage(content="You are an expert market analyst providing professional research reports."),
                HumanMessage(content=analysis_prompt)
            ])
            
            results['final_analysis'] = response.content
            results['sources'] = all_urls
            results['total_sources'] = len(all_urls)
            
        except Exception as e:
            results['final_analysis'] = f"Analysis error: {str(e)}"
        
        return results

# ------- UI COMPONENTS -------
def render_header():
    """Render professional header"""
    st.markdown("""
    <div class="main-header">
        <h1>🔍 AI Research Agent</h1>
        <p>Autonomous multi-agent research system powered by LangGraph & GPT-4</p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.85;">
            Built by <a href="https://linkedin.com/in/lokesh-k-7b891b216" target="_blank" 
            style="color: white; text-decoration: underline; font-weight: 600;">Lokesh Kaluvala</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render configuration sidebar"""
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # API Keys
        with st.expander("🔑 API Keys", expanded=not st.session_state.api_keys_set):
            openai_key = st.text_input(
                "OpenAI API Key",
                type="password",
                help="Get from https://platform.openai.com/api-keys"
            )
            tavily_key = st.text_input(
                "Tavily API Key",
                type="password",
                help="Get from https://tavily.com"
            )
            
            if st.button("Save Keys"):
                if openai_key and tavily_key:
                    st.session_state.openai_key = openai_key
                    st.session_state.tavily_key = tavily_key
                    st.session_state.api_keys_set = True
                    st.success("✅ Keys saved!")
                    st.rerun()
                else:
                    st.error("Please enter both keys")
        
        st.markdown("---")
        
        # Example queries
        st.markdown("### 💡 Example Queries")
        examples = [
            "NVIDIA Blackwell GPU demand 2026",
            "Tesla Cybertruck production forecast",
            "Apple Vision Pro market adoption",
            "AMD MI300 vs NVIDIA H100 comparison",
            "Global semiconductor shortage outlook"
        ]
        
        for example in examples:
            if st.button(example, key=f"ex_{example}", use_container_width=True):
                st.session_state.selected_example = example
                st.rerun()
        
        st.markdown("---")
        
        # Stats
        if st.session_state.research_history:
            st.markdown("### 📊 Statistics")
            st.metric("Total Searches", len(st.session_state.research_history))
            
            total_sources = sum(r.get('total_sources', 0) for r in st.session_state.research_history)
            st.metric("Sources Analyzed", total_sources)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        **Built by:** [Lokesh Kaluvala](https://linkedin.com/in/lokesh-k-7b891b216)
        
        **Tech Stack:**
        - 🦜 LangChain & LangGraph
        - 🤖 OpenAI GPT-4
        - 🔍 Tavily Search API
        - 🎨 Streamlit
        
        **Features:**
        - Multi-iteration research
        - Source verification
        - Structured analysis
        - Real-time progress
        
        **Connect:**
        - [💼 LinkedIn](https://linkedin.com/in/lokesh-k-7b891b216)
        - [💻 GitHub](https://github.com/lokesh-kaluvala)
        """)

def render_research_form():
    """Render main research input form"""
    st.markdown("### 🎯 Start Your Research")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        default_query = st.session_state.get('selected_example', '')
        query = st.text_input(
            "Research Query",
            placeholder="e.g., NVIDIA Blackwell GPU demand and supply 2026",
            value=default_query,
            label_visibility="collapsed"
        )
        if 'selected_example' in st.session_state:
            del st.session_state.selected_example
    
    with col2:
        search_button = st.button("🚀 Research", type="primary", use_container_width=True)
    
    return query, search_button

def render_research_progress(progress_placeholder):
    """Render research progress"""
    with progress_placeholder:
        st.markdown('<div class="processing">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-value">🔍</div>
                <div class="metric-label">Researching</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-value">📊</div>
                <div class="metric-label">Analyzing</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-value">⚡</div>
                <div class="metric-label">Processing</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

def render_results(results: Dict):
    """Render research results"""
    st.markdown("---")
    st.markdown("## 📊 Research Results")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Iterations", len(results['iterations']))
    
    with col2:
        st.metric("Sources Found", results['total_sources'])
    
    with col3:
        st.metric("Timestamp", results['timestamp'].split()[1])
    
    with col4:
        confidence = "HIGH" if results['total_sources'] >= 8 else "MEDIUM" if results['total_sources'] >= 5 else "LOW"
        st.metric("Confidence", confidence)
    
    # Main analysis
    st.markdown("### 📝 Analysis Report")
    st.markdown(results['final_analysis'])
    
    # Sources
    if results['sources']:
        with st.expander(f"📚 View All Sources ({len(results['sources'])})"):
            for i, url in enumerate(results['sources'], 1):
                st.markdown(f"""
                <div class="source-link">
                    <strong>{i}.</strong> <a href="{url}" target="_blank">{url}</a>
                </div>
                """, unsafe_allow_html=True)
    
    # Iteration details
    with st.expander("🔄 Research Iterations Details"):
        for iteration in results['iterations']:
            st.markdown(f"""
            **Iteration {iteration['iteration']}**
            - Query: `{iteration['query']}`
            - Sources: {iteration.get('sources_found', 'N/A')}
            """)
    
    # Export options
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        # Download as text
        report_text = f"""
AI RESEARCH REPORT
==================
Query: {results['query']}
Timestamp: {results['timestamp']}
Total Sources: {results['total_sources']}

ANALYSIS
========
{results['final_analysis']}

SOURCES
=======
{chr(10).join(f"{i}. {url}" for i, url in enumerate(results['sources'], 1))}
        """
        
        st.download_button(
            "📥 Download Report (TXT)",
            report_text,
            file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    with col2:
        if st.button("🔄 New Research"):
            st.session_state.current_research = None
            st.rerun()

# ------- MAIN APP -------
def main():
    init_session_state()
    
    render_header()
    render_sidebar()
    
    # Check API keys
    if not st.session_state.api_keys_set:
        st.info("👈 Please configure your API keys in the sidebar to get started")
        
        st.markdown("""
        ### 🚀 Getting Started
        
        1. **Get API Keys:**
           - OpenAI: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
           - Tavily: [tavily.com](https://tavily.com) (Free tier: 1000 searches/month)
        
        2. **Enter keys** in the sidebar
        
        3. **Start researching!** Try example queries or create your own
        
        ### ✨ Features
        - 🔄 Multi-iteration research for comprehensive coverage
        - 📊 Structured analysis with bull/bear cases
        - 🔗 Source tracking and verification
        - 📥 Export reports for sharing
        - ⚡ Real-time progress updates
        """)
        return
    
    # Main research interface
    query, search_button = render_research_form()
    
    if search_button and query:
        if len(query) < 10:
            st.error("Please enter a more detailed query (at least 10 characters)")
            return
        
        # Execute research
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        render_research_progress(progress_placeholder)
        
        try:
            agent = StreamlitResearchAgent(
                st.session_state.openai_key,
                st.session_state.tavily_key
            )
            
            def update_status(message):
                status_placeholder.info(message)
            
            results = agent.research(query, progress_callback=update_status)
            
            # Clear progress
            progress_placeholder.empty()
            status_placeholder.empty()
            
            # Store and display results
            st.session_state.current_research = results
            st.session_state.research_history.append(results)
            
            st.success("✅ Research completed successfully!")
            render_results(results)
            
        except Exception as e:
            progress_placeholder.empty()
            status_placeholder.empty()
            st.error(f"❌ Research failed: {str(e)}")
            st.info("Please check your API keys and try again")
    
    elif st.session_state.current_research:
        render_results(st.session_state.current_research)
    
    # Footer with your info
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #64748b; padding: 2rem 0;">
        <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">Built with ❤️ using LangGraph, OpenAI GPT-4, and Streamlit</p>
        <p style="font-size: 1rem; font-weight: 600; color: #475569; margin: 0.5rem 0;">
            Project by <a href="https://linkedin.com/in/lokesh-k-7b891b216" target="_blank" style="color: #6366f1; text-decoration: none; font-weight: 700;">Lokesh Kaluvala</a>
        </p>
        <p style="font-size: 0.875rem; margin-top: 0.5rem;">
            <a href="https://linkedin.com/in/lokesh-k-7b891b216" target="_blank" style="color: #6366f1; text-decoration: none; margin: 0 0.75rem;">
                💼 LinkedIn
            </a>
            <a href="https://github.com/lokesh-kaluvala" target="_blank" style="color: #6366f1; text-decoration: none; margin: 0 0.75rem;">
                💻 GitHub
            </a>
        </p>
        <p style="font-size: 0.875rem; color: #94a3b8; margin-top: 1rem;">© 2026 AI Research Agent | Version 1.0</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()