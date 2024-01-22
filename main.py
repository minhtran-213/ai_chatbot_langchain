from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PythonLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from langchain import hub
from langchain.chains.question_answering import load_qa_chain


def load_llm(model):
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    return LlamaCpp(
        model_path=model,
        n_ctx=5000,
        n_gpu_layers=1,
        n_batch=512,
        f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
        callback_manager=callback_manager,
        verbose=True,
    )


def create_db_from_files(director_path):
    # Khai bao loader de quet toan bo thu muc dataa
    loader = DirectoryLoader(director_path, glob="*.py", loader_cls=PythonLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Embeding
    embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")
    local_db = FAISS.from_documents(chunks, embedding_model)
    local_db.save_local(vector_db_path)
    return local_db


# Tao prompt template
def creat_prompt():
    custom_template = """
        You're an experienced software engineer.
        Your expertise is software architecture, and Python language.
        You have worked with FastAPI for many years.
        
        Your task is to look at the project context bellow and then answer the questions that need to improve the code.
        You should follow the convention from Python community.
        {context}
        Questions: {question}
        
        Please, make sure you keep the code clean. If you don't know, just say that you don't know. Don't try to make up the answer
    """
    return PromptTemplate(
        input_variables=["context", "question"],
        template=custom_template,
    )


def create_rag_default_prompt():
    return hub.pull("rlm/rag-prompt-default")


def get_retriever(db_local):
    return db_local.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8}
    )


def create_qa_chain(prompt_default, llm, db_local):
    retriever = get_retriever(db_local)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={'prompt': prompt_default}
    )


def generate_qa_chain(llm, prompt):
    return load_qa_chain(llm, chain_type="stuff", prompt=prompt)


# Read tu VectorDB
def read_vectors_db():
    embedding_model = GPT4AllEmbeddings(model_file="model/all-MiniLM-L6-v2-f16.gguf")
    return FAISS.load_local(vector_db_path, embedding_model)


if __name__ == '__main__':
    model_file = "/home/minhtranb/works/personal/ai_tools/qabot_for_code/model/codellama-13b-instruct.Q4_K_M.gguf"
    vector_db_path = "vectorstores/db_faiss"
    work_path = "/home/minhtranb/works/personal/ai_tools/qabot_for_code/"
    create_db_from_files(work_path)
    db = read_vectors_db()

    llm = load_llm(model_file)
    prompt = creat_prompt()
    question = "Which method is the is used to load qa chain?"
    docs = db.search(question, search_type="similarity")
    llm_chain = generate_qa_chain(llm, prompt)
    response = llm_chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    print(response)
