import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from langchain_community.document_loaders import TextLoader

# from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# from langchain_community.document_loaders import UnstructuredMarkdownLoader
# from langchain_text_splitters import CharacterTextSplitter
# from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain

from langchain_community.llms import Ollama
# from langchain_ollama.llms import OllamaLLM


from langchain_chroma import Chroma

# from langchain.chains import LLMChain
# from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# from langchain_core.runnables import RunnablePassthrough


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing_extensions import TypedDict
from typing import List
from langgraph.graph import END, StateGraph
import pydantic

llm = Ollama(model="llama3.2", base_url="http://localhost:11434/")


embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")

# 使用模型
embed_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-zh-v1.5",
    cache_folder="models/embeddings/",
    encode_kwargs={"normalize_embeddings": True},
)


# TXT Data preprocess
loader = TextLoader("paper_sorting.txt")
TXT_data = loader.load()

# text_splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=500)
all_splits = text_splitter.split_documents(TXT_data)

# load vectordb
db = Chroma.from_documents(documents=all_splits, embedding=embed_model)

# retrieve
retriever = db.as_retriever()


class filesearch(BaseModel):
    """
    查詢提供的文件
    """

    # Pydantic: Field()-> 詳細的說明，用於資料驗證
    query: str = Field(description="搜尋向量資料庫時輸入的問題")


# RAG Model
def RAGModel():
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
    return rag_chain



# LLM Model
def LLM():
    # 輸出格式
    template = """
        負責將傳入的問題整理成語句通順的疑問句，用中文回答
        問題: {question}
        """

    prompt = ChatPromptTemplate.from_template(template)

    # 套用語言模型與輸出格式
    # fmt: off
    llm_chain = (
        {"question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    # fmt: on
    return llm_chain




# 判斷文件與問題的相關性
class IsREL(BaseModel):
    """
    確認提取文章與問題是否有關(相關 or 不相關)
    """

    # Pydantic: Field()-> 詳細的說明，用於資料驗證
    binary_score: str = Field(
        description="請問文章與問題是否相關。('相關' or '不相關')"
    )


def Retrieval_demand():
    instruction = """
                    你是一個評分人員，負責評估文件內容與問題的關聯性。
                    輸出'相關' or '不相關'
                    """
    Retrieval_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", instruction),
            ("human", "文件內容: \n\n {document} \n\n 問題: {question}"),
        ]
    )

    Retrieval_prompt = Retrieval_prompt | llm

    # Grader LLM
    llm_grader = llm.with_structured_output(IsREL)

    # 使用 LCEL 語法建立 chain
    retrieval_demand = Retrieval_prompt | llm_grader
    return retrieval_demand


# 判斷語言模型生成的答案是否有幻覺(有沒有事實能佐證)
class IsSUP(BaseModel):
    """
    確認答案是否為虛構('虛構的' or '基於文件內容得出' or '一半虛構一半由文件得出')
    """

    binary_score: str = Field(
        description="答案是否由為虛構。('生成的答案是虛構的' or '生成的答案是是基於文件內容得出')"  # noqa: E501
    )


def supported():
    instruction = """
    你是一個評分的人員，負責確認LLM生成的答案是否為虛構的。
    以下會給你一個文件與相對應的LLM生成的答案，
    請輸出 '生成的答案是虛構的' or '生成的答案是是基於文件內容得出'做為判斷結果。
    """
    supported_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", instruction),
            ("human", "取用文件: \n\n {documents} \n\n 回答: {generation}"),
        ]
    )

    supported_prompt = supported_prompt | llm

    llm_grader = llm.with_structured_output(IsSUP)

    # 使用 LCEL 語法建立 chain
    supported_grader = supported_prompt | llm_grader
    return supported_grader


# 判斷語言模型生成的答案是否可以正確回答使用者的問題
class IsUSE(BaseModel):
    """
    確認答案是否可以正確回答使用者的問題
    """

    # Pydantic: Field()-> 詳細的說明，用於資料驗證
    binary_score: str = Field(
        description="是否可以正確回答使用者的問題。('有回應到問題' or '沒有回應到問題')"
    )


def response():
    instruction = """
                你是一個評分人員，確認回答的答案是否回應問題，
                輸出 '有回應到問題' or '沒有回應到問題'。
                """
    response_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", instruction),
            ("human", "問題: \n\n {question} \n\n 答案: {generation}"),
        ]
    )

    response_prompt = response_prompt | llm
    llm_grader = llm.with_structured_output(IsUSE)

    # 使用 LCEL 語法建立 chain
    response_grader = response_prompt | llm_grader
    return response_grader


# Graph_RAG 設定
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]


# retrieve
def retrieve(state):
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)

    return {"documents": documents, "question": question}


# 判斷問題是否與文件有關
def retrieval_demand(state):
    documents = state["documents"]
    question = state["question"]

    filtered_docs = []
    retrieval_gra = Retrieval_demand()

    for d in documents:
        score = retrieval_gra.invoke({"question": question, "document": d.page_content})
        grade = score.binary_score
        if grade == "yes":
            filtered_docs.append(d)
        else:
            continue
    return {"documents": filtered_docs, "question": question}


# RAG
def rag(state):
    question = state["question"]
    # documents = state["documents"]

    rag_chain = RAGModel()

    # RAG
    # generation = rag_chain.invoke({"question": question})
    # generation = rag_chain.invoke(question)
    generation = rag_chain.invoke(question)
    return {"question": question, "generation": generation}


# LLM
def answer(state):
    question = state["question"]
    llm_chain = LLM()

    # generation = llm_chain.predict(human_input=question)
    generation = llm_chain.invoke({"question": question})
    return {"question": question, "generation": generation}


# 針對Graph State中的內容進行判斷決定流程後續進入到哪個Node中
def route_question(state):
    question = state["question"]
    RAG = RAGModel()

    # source = RAG.invoke({"question": question})
    source = RAG.invoke(question)
    # source = RAG.invoke({"documents": retriever, "question": question})

    if "tool_calls" not in source:
        return "answer"
    if len(source["tool_calls"]) == 0:
        raise Exception("Router could not decide source")

    datasource = source["tool_calls"][0]["function"]["name"]
    if datasource == "web_search":
        return "web_search"
    elif datasource == "filesearch":
        return "filesearch"


def route_retrieval(state):
    filtered_documents = state["documents"]

    if not filtered_documents:
        return "web_search"
    else:
        return "rag"


def grade_rag_generation(state):
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    hallucination_grader = supported()

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score

    answer_grader = response()

    # 確認有無幻覺
    if grade == "no":
        # 檢查答案符不符合問題
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            return "useful"
        else:
            return "not useful"
    else:
        return "not supported"


# Graph_RAG
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("retrieval_demand", retrieval_demand)  # retrieval demand
workflow.add_node("rag", rag)  # rag
workflow.add_node("answer", answer)  # llm

# 建構
workflow.set_conditional_entry_point(
    route_question,
    {
        "filesearch": "retrieve",
        "answer": "answer",
    },
)
workflow.add_edge("retrieve", "retrieval_demand")
workflow.add_conditional_edges(
    "retrieval_demand",
    route_retrieval,
    {
        "rag": "rag",
    },
)
workflow.add_conditional_edges(
    "rag",
    grade_rag_generation,
    {
        "not supported": "rag",
        "useful": END,
    },
)
workflow.add_edge("answer", END)

# Compile
app = workflow.compile()


# Graph_RAG(內含 Self_RAG)
def Graph_and_Self_RAG(question: str) -> str | None:
    inputs = {"question": question}
    # 取出最後一筆輸出作為output
    for output in app.stream(inputs):
        print("\n")

    if "rag" in output.keys():
        return output["rag"]["generation"]

    elif "answer" in output.keys():
        return output["answer"]["generation"]
