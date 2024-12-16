import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

llm = Ollama(model="llama3.2")


# 使用模型
embed_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-zh-v1.5",
    cache_folder="models/embeddings/",
    encode_kwargs={"normalize_embeddings": True},
)

def retrieve(paper_sorting:str):
    # TXT Data preprocess
    loader = TextLoader(paper_sorting)
    TXT_data = loader.load()

    # text_splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=500)
    all_splits = text_splitter.split_documents(TXT_data)

    # load vectordb
    db = Chroma.from_documents(documents=all_splits, embedding=embed_model)

    # retrieve
    retriever = db.as_retriever()
    return retriever


# RAG Model
def RAGModel(question:str, retriever):
    template = """僅根據以下上下文回答問題:
        (回答時不需加入根據，直接回答答案)
        {context}

        問題: {question}
        """

    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    )
    generation = rag_chain.invoke(question)
    return generation

que = input("輸入想要查詢的關鍵字：")
ret = retrieve(paper_sorting="paper_sorting.txt")
Answer = RAGModel(question=que, retriever=ret)



