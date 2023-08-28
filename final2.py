import streamlit as st

import urllib.request
import os
import re
import openai

from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains import RetrievalQA
os.environ["OPENAI_API_KEY"] = "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"


st.set_page_config(page_title="CHECK DETAILS FROM YOUR RESUME")
custom="""
<style>
div.css-1kyxreq.e115fcil2 
{
width: 50%;
margin: auto;
}


div.css-1v0mbdj.e115fcil1 
{
    
    margin:auto;
}

div.block-container.css-1y4p8pa.ea3mdgi4
{
    margin-top: 5%;


}


span.css-10trblm.e1nzilvr0
{
    line-height: 76.8px;
    color: #192E48;
    font-size: 64px;
    font-weight: 700 !important;
    box-sizing: border-box;
    font-family: 'Exo 2', sans-serif;
    margin-bottom: 1rem !important;
    white-space: nowrap;
}



div.css-5rimss.e1nzilvr4
{
    margin-left: -12%;
}


div.css-16idsys.e1nzilvr4   
{
    
    margin-left: 22%;
}
div.css-16idsys.e1nzilvr4    > p
{
    
    font-size: 32px;
    line-height: 41.6px;
    font-weight: 600 !important;
    font-family: 'Exo 2', sans-serif;
}


div.st-b3.st-b8.st-bv.st-b1.st-bn.st-ae.st-af.st-ag.st-ah.st-ai.st-aj.st-bw.st-bs > input.st-bc.st-bx.st-by.st-bz.st-c0.st-c1.st-c2.st-c3.st-c4.st-c5.st-c6.st-b8.st-c7.st-c8.st-c9.st-ca.st-cb.st-cc.st-cd.st-ce.st-ae.st-af.st-ag.st-cf.st-ai.st-aj.st-bw.st-cg.st-ch.st-ci
{
	border: 1px solid rgba(39, 71, 110, 0.16);
    border-radius: 8px;
    padding: 8px 40px;
    height: 53px;
    width: 100%;
    font-size: var(--font_16);
    line-height: var(--font_16_line-height);
    color: #999999;
    font-size: 16px;
    line-height: 24px;
    font-family: 'Exo 2', sans-serif;
    font-weight: 400;
}

div.st-bc.st-b3.st-bd.st-b8.st-be.st-bf.st-bg.st-bh.st-bi.st-bj.st-bk.st-bl.st-bm.st-b1.st-bn.st-au.st-ax.st-av.st-aw.st-ae.st-af.st-ag.st-ah.st-ai.st-aj.st-bo.st-bp.st-bq.st-br.st-bs.st-bt.st-bu
{
    border-radius: 8px;
    padding: 16px;
    box-sizing: border-box;
    font-family: 'Exo 2', sans-serif;
    font-size: 1rem;
    font-weight: 400;
    line-height: 1.5;
    width: 150%;
    margin-left: -25%;
    border: 1px solid rgba(39, 71, 110, 0.16);
    background-color: #fff;
    
}

div.block-container.css-1y4p8pa.ea3mdgi4
{
    margin-top: -2%;
}
<style>
"""

st.markdown(f"<style>{custom}</style>", unsafe_allow_html=True)



image_url = 'https://findyourvisa.com/images/logo.png'
st.image(image_url)

st.header("Finding visa, simplified.")


# upload file
pdf = "sodapdf-converted-9.pdf"
    
    # extract the text
if pdf is not None:
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
        
      # split into chunks
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
    )
chunks = text_splitter.split_text(text)

embeddings = OpenAIEmbeddings()
knowledge_base = FAISS.from_texts(chunks, embeddings)



      
user_question = st.text_input("Not sure? Search here.")
if user_question:
    docs = knowledge_base.similarity_search(user_question)
            
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff")
            

    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=user_question)
        print(cb)
              
    st.write(response)