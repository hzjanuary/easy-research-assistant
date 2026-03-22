"""Streamlit Chat Interface for easyResearchAssistant."""
import os
import json
import requests
import streamlit as st
from typing import Generator

# API Configuration
API_BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
DEFAULT_ACCESS_TOKEN = os.getenv("ACCESS_TOKEN", "")

st.set_page_config(
    page_title="easyResearchAssistant",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .stChatMessage { background-color: transparent; }
    .research-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 4px 12px; border-radius: 16px;
        font-size: 0.8rem; font-weight: 600;
    }
    .status-online { color: #10b981; font-weight: 600; }
    .status-offline { color: #ef4444; font-weight: 600; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "access_token" not in st.session_state:
        st.session_state.access_token = DEFAULT_ACCESS_TOKEN
    
    if "research_mode" not in st.session_state:
        st.session_state.research_mode = False
    
    if "api_connected" not in st.session_state:
        st.session_state.api_connected = False
    
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7
    
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = 2048


def check_api_health() -> dict:
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"status": "error", "message": f"HTTP {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"status": "error", "message": "Connection refused"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def stream_chat_response(
    prompt: str,
    token: str,
    research_mode: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 2048
) -> Generator[str, None, None]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "prompt": prompt,
        "stream": True,
        "research_mode": research_mode,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        with requests.post(
            f"{API_BASE_URL}/v1/inference",
            headers=headers,
            json=payload,
            stream=True,
            timeout=120
        ) as response:
            if response.status_code != 200:
                yield f"Error: HTTP {response.status_code}"
                return
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith("data: "):
                        data = line_str[6:]
                        
                        if data == "[DONE]":
                            return
                        
                        try:
                            parsed = json.loads(data)
                            
                            if "info" in parsed:
                                yield f"\n*{parsed['info']}*\n"
                                continue
                            
                            if "error" in parsed:
                                yield f"\n**Error:** {parsed['error']}"
                                return
                            
                            if "response" in parsed:
                                response_text = parsed["response"]
                                if response_text is not None:
                                    yield str(response_text)
                            
                        except json.JSONDecodeError:
                            yield data
                            
    except requests.exceptions.Timeout:
        yield "Error: Request timed out. Please try again."
    except requests.exceptions.ConnectionError:
        yield "Error: Cannot connect to the API gateway."
    except Exception as e:
        yield f"Error: {str(e)}"


def render_sidebar():
    with st.sidebar:
        st.title("Settings")
        
        st.subheader("Authentication")
        token = st.text_input(
            "Access Token",
            value=st.session_state.access_token,
            type="password",
            help="Your access token for the API"
        )
        if token != st.session_state.access_token:
            st.session_state.access_token = token
        
        st.divider()
        
        st.subheader("Research Mode")
        research_mode = st.toggle(
            "Enable Research Mode",
            value=st.session_state.research_mode,
            help="Searches the web in real-time for up-to-date information (RAG)"
        )
        st.session_state.research_mode = research_mode
        
        if research_mode:
            st.info(
                "🔍 **Research Mode active**: The assistant will search the web "
                "for real-time information before responding, ensuring up-to-date answers."
            )
        
        st.divider()
        
        st.subheader("Parameters")
        
        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.temperature,
            step=0.1,
            help="Higher = more creative, Lower = more focused"
        )
        
        st.session_state.max_tokens = st.slider(
            "Max Tokens",
            min_value=256,
            max_value=4096,
            value=st.session_state.max_tokens,
            step=256,
            help="Maximum response length"
        )
        
        st.divider()
        
        st.subheader("Gateway Status")
        
        health = check_api_health()
        if health.get("status") == "healthy":
            st.session_state.api_connected = True
            st.markdown('<span class="status-online">● Online</span>', unsafe_allow_html=True)
            
            providers = health.get("providers", {})
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Cloud Nodes", f"{providers.get('cloud_available', 0)}/{providers.get('cloud_total', 0)}")
            with col2:
                fallback = "Yes" if providers.get("local_fallback") else "No"
                st.metric("Local Fallback", fallback)
        else:
            st.session_state.api_connected = False
            st.markdown('<span class="status-offline">● Offline</span>', unsafe_allow_html=True)
            st.error(health.get("message", "Connection failed"))
        
        st.divider()
        
        st.subheader("Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        
        with col2:
            if st.button("Refresh", use_container_width=True):
                st.rerun()


def render_chat():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("easyResearchAssistant")
        st.caption("High-Availability AI Inference Gateway")
    
    with col2:
        if st.session_state.research_mode:
            st.markdown(
                '<span class="research-badge">🔍 Research Mode (RAG)</span>',
                unsafe_allow_html=True
            )
    
    if not st.session_state.api_connected:
        st.warning(
            "Not connected to the API gateway. "
            "Make sure the server is running: `python api_gateway.py`"
        )
    
    if not st.session_state.access_token:
        st.info("Enter your access token in the sidebar to start chatting.")
    
    # Chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input(
        "Enter your message...",
        disabled=not (st.session_state.api_connected and st.session_state.access_token)
    ):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            # Show search status if research mode is enabled
            if st.session_state.research_mode:
                with st.status("🔍 Searching the web for latest info...", expanded=False) as search_status:
                    st.write("Querying DuckDuckGo for real-time information...")
                    
                    # Start streaming - the search happens on backend
                    for chunk in stream_chat_response(
                        prompt=prompt,
                        token=st.session_state.access_token,
                        research_mode=st.session_state.research_mode,
                        temperature=st.session_state.temperature,
                        max_tokens=st.session_state.max_tokens
                    ):
                        if chunk is not None:
                            chunk_text = chunk if isinstance(chunk, str) else str(chunk)
                            if not chunk_text:
                                continue
                            # Update status once we start getting response
                            if not full_response:
                                search_status.update(label="✅ Search complete - generating response...", state="complete")
                            full_response += chunk_text
                            response_placeholder.markdown(full_response + "▌")
            else:
                for chunk in stream_chat_response(
                    prompt=prompt,
                    token=st.session_state.access_token,
                    research_mode=st.session_state.research_mode,
                    temperature=st.session_state.temperature,
                    max_tokens=st.session_state.max_tokens
                ):
                    if chunk is not None:
                        chunk_text = chunk if isinstance(chunk, str) else str(chunk)
                        if not chunk_text:
                            continue
                        full_response += chunk_text
                        response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})


def main():
    init_session_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
