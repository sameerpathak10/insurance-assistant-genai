# The below frontend code is provided by AWS and Streamlit. 

import streamlit as st
import rag_backend as demo 

st.set_page_config(page_title="Insurance Assistant with RAG")

new_title='<p style="font-family:sans-serif; color:Gree; font-size: 42px;">Insurance Assistant with RAG </p>'
st.markdown(new_title, unsafe_allow_html=True)

if 'vector_index' not in st.session_state:
    with st.spinner("Wait for sometime. Creating Vector store takes time. Approximate time 15 mins :-"):
        st.session_state.vector_index = demo.create_index()

input_text = st.text_area("Input text", label_visibility="collapsed")
go_button = st.button("Generate Prompt", type="primary")


if go_button:
    st.write(f"Your prompt: {input_text}")
    with st.spinner("Generating response..."):
        response_content=demo.get_rag_response(index=st.session_state.vector_index, question=input_text)
        st.write(response_content)

footer='<footer style="font-family:sans-serif; color:Gree; font-size: 10px;">Developed by Sameer Pathak </footer>'
st.markdown(footer, unsafe_allow_html=True)        