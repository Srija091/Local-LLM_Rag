import streamlit as st
import ollama
from rag import load_pdf
# Streamlit App
st.set_page_config(page_title="My Local LLM", layout="wide")
st.title("My Local LLM Interface")
st.caption("ask anything to your local LLM model - NO DATA LEAVES YOUR MACHINE")
#memory for LLM 
if "messages" not in st.session_state:
    st.session_state["messages"]=[{"role":"assistant","content":"how can I help you today?"}]
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"]=None
# PDF Upload and RAG
with st.sidebar:
    st.header("PDF RAG")
    uploaded_file = st.file_uploader("upload your PDF",type=["pdf"])
    if uploaded_file:
        with open("temp.pdf","wb") as f:
            f.write(uploaded_file.read())
        st.session_state.vectorstore = load_pdf("temp.pdf")
        st.success("PDF indexed successfully!")
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = [{"role":"assistant","content":"how can i help you today?"}]
        st.rerun()
# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
#interaction LOOP
if prompt := st.chat_input("whats on your input?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    context = ""
    if st.session_state.vectorstore:
        docs = st.session_state.vectorstore.similarity_search(prompt,k=10)
        context = "\n\n".join(d.page_content for d in docs)
    # Prepare messages for Ollama LLM
    messages = [
        {"role": "system", "content": f"You are a helpful assistant. Use the following PDF content to answer the question:\n\n{context}"},
        {"role": "user", "content": prompt}
    ]

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        stream = ollama.chat(
            model='llama3.2',
            messages = messages,
            stream = True,
        )
        for chunk in stream:
            if chunk['message']['content']:
                content = chunk['message']['content']
                full_response += content
                response_placeholder.markdown(full_response + "▌")
        
        # Final update to remove the cursor
        response_placeholder.markdown(full_response)
    
    # 5. Save the AI's response to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})


