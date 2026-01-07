"""
Multi-Agent Research System with Self-Correction
Python 3.11+ compatible
"""

import os
from typing import TypedDict, List, Annotated
import operator
from datetime import datetime

# Core dependencies
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage

# --- 1. CONFIGURATION ---
class Config:
    """Centralized configuration"""
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    MODEL = "gpt-4o-mini"  # More cost-effective, still powerful
    TEMPERATURE = 0.3  # Slight creativity for better synthesis
    MAX_REVISIONS = 3
    SEARCH_RESULTS = 10
    
    @classmethod
    def validate(cls):
        """Ensure API keys are set"""
        if not cls.TAVILY_API_KEY:
            raise ValueError("TAVILY_API_KEY not set. Get one at https://tavily.com")
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set")


# --- 2. ENHANCED STATE MANAGEMENT ---
class AgentState(TypedDict):
    """Enhanced state with better tracking"""
    query: str
    refined_query: str
    raw_data: Annotated[List[str], operator.add]  # Accumulate across iterations
    source_urls: Annotated[List[str], operator.add]
    analysis: str
    revision_count: int
    is_satisfactory: bool
    feedback: str  # Track why we're iterating
    timestamp: str


# --- 3. INTELLIGENT AGENT NODES ---
class ResearchAgent:
    """Improved research orchestration"""
    
    def __init__(self):
        Config.validate()
        self.llm = ChatOpenAI(
            model=Config.MODEL, 
            temperature=Config.TEMPERATURE,
            request_timeout=30
        )
        self.search_tool = TavilySearchResults(
            max_results=Config.SEARCH_RESULTS,
            search_depth="advanced"  # Better quality results
        )
    
    def researcher_node(self, state: AgentState) -> dict:
        """Enhanced researcher with query refinement"""
        revision = state.get("revision_count", 0)
        
        # Progressive query refinement based on feedback
        if revision == 0:
            refined = f"{state['query']} latest news 2026 market analysis"
        elif revision == 1:
            refined = f"{state['query']} expert opinions forecasts 2026"
        else:
            refined = f"{state['query']} {state.get('feedback', '')} detailed report"
        
        print(f"\n🔍 RESEARCH ITERATION {revision + 1}")
        print(f"Query: {refined}")
        
        try:
            results = self.search_tool.invoke(refined)
            
            cleaned_data = []
            urls = []
            
            for r in results:
                content = r.get('content', '').strip()
                url = r.get('url', '')
                
                # Filter out low-quality results
                if len(content) > 50 and not content.startswith('Error'):
                    cleaned_data.append(content)
                    urls.append(url)
            
            print(f"✓ Found {len(cleaned_data)} quality sources")
            
            return {
                "refined_query": refined,
                "raw_data": cleaned_data,
                "source_urls": urls,
                "revision_count": revision + 1,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Research error: {e}")
            return {
                "raw_data": [f"Search failed: {str(e)}"],
                "source_urls": [],
                "revision_count": revision + 1
            }
    
    def analyst_node(self, state: AgentState) -> dict:
        """Enhanced analyst with structured output"""
        
        # Combine all accumulated data
        all_data = "\n\n---\n\n".join(state.get("raw_data", []))
        
        system_prompt = """You are an expert financial analyst specializing in technology markets.
Your goal is to provide actionable intelligence from research data."""
        
        analysis_prompt = f"""
Analyze the following research on: {state['query']}

DATA SOURCES ({len(state.get('raw_data', []))} total):
{all_data[:8000]}  # Limit context to avoid token limits

REQUIREMENTS:
1. **Key Players**: List 3-5 major companies/entities (ignore fragments like "Now Generally")
2. **Bull Case 2026**: What factors support growth/success?
3. **Bear Case 2026**: What are the risks/challenges?
4. **Data Quality**: Are there contradictions or gaps? (e.g., conflicting dates, insufficient details)
5. **Confidence**: Rate your analysis confidence (HIGH/MEDIUM/LOW)

If the data is clearly insufficient, outdated, or contradictory, respond with:
"INSUFFICIENT_DATA: [specific reason]"

Otherwise, provide a structured analysis.
"""
        
        print("\n🧠 ANALYZING DATA...")
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=analysis_prompt)
            ]
            response = self.llm.invoke(messages)
            analysis = response.content
            
            # Check quality
            is_insufficient = "INSUFFICIENT_DATA" in analysis
            is_short = len(analysis) < 200
            is_satisfactory = not (is_insufficient or is_short)
            
            # Generate feedback for next iteration
            feedback = ""
            if is_insufficient:
                feedback = "more detailed current data needed"
            elif is_short:
                feedback = "deeper market analysis required"
            
            print(f"✓ Analysis complete - Satisfactory: {is_satisfactory}")
            
            return {
                "analysis": analysis,
                "is_satisfactory": is_satisfactory,
                "feedback": feedback
            }
            
        except Exception as e:
            print(f"❌ Analysis error: {e}")
            return {
                "analysis": f"Analysis failed: {str(e)}",
                "is_satisfactory": False,
                "feedback": "retry with different approach"
            }


# --- 4. WORKFLOW ORCHESTRATION ---
def build_workflow():
    """Construct the agent workflow graph"""
    
    agent = ResearchAgent()
    
    def should_continue(state: AgentState) -> str:
        """Smart routing with max iteration safety"""
        if state["is_satisfactory"]:
            print("\n✅ Analysis satisfactory - completing")
            return "end"
        
        if state["revision_count"] >= Config.MAX_REVISIONS:
            print(f"\n⚠️  Max revisions ({Config.MAX_REVISIONS}) reached - completing")
            return "end"
        
        print(f"\n🔄 Iteration needed: {state.get('feedback', 'improving research')}")
        return "research"
    
    # Build graph
    workflow = StateGraph(AgentState)
    
    workflow.add_node("research", agent.researcher_node)
    workflow.add_node("analyze", agent.analyst_node)
    
    workflow.add_edge(START, "research")
    workflow.add_edge("research", "analyze")
    workflow.add_conditional_edges(
        "analyze",
        should_continue,
        {
            "research": "research",
            "end": END
        }
    )
    
    return workflow.compile()


# --- 5. EXECUTION & FORMATTING ---
def run_research(query: str):
    """Execute research with formatted output"""
    
    print("="*60)
    print(f"🚀 STARTING RESEARCH AGENT")
    print(f"Query: {query}")
    print("="*60)
    
    app = build_workflow()
    
    initial_state = {
        "query": query,
        "raw_data": [],
        "source_urls": [],
        "revision_count": 0,
        "is_satisfactory": False,
        "feedback": "",
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        final_state = None
        for output in app.stream(initial_state):
            final_state = output
        
        # Extract final state
        if final_state:
            result = list(final_state.values())[0]
            
            print("\n" + "="*60)
            print("📊 FINAL RESEARCH REPORT")
            print("="*60)
            print(f"\n{result.get('analysis', 'No analysis generated')}")
            print(f"\n\n📚 SOURCES ({len(result.get('source_urls', []))}):")
            for i, url in enumerate(result.get('source_urls', [])[:5], 1):
                print(f"{i}. {url}")
            print("\n" + "="*60)
            
            return result
        
    except Exception as e:
        print(f"\n❌ EXECUTION ERROR: {e}")
        raise


# --- 6. MAIN ENTRY POINT ---
if __name__ == "__main__":
    # Example queries
    EXAMPLE_QUERIES = [
        "NVIDIA Blackwell GPU demand and supply 2026",
        "Apple Vision Pro market adoption 2026",
        "Tesla Cybertruck production ramp 2026"
    ]
    
    # Run the research
    query = EXAMPLE_QUERIES[0]  # Change index to try different queries
    run_research(query)