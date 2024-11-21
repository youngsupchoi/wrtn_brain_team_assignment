import os
from IPython import embed
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
# from langchain.retrievers import ContextualCompressionRetriever
from kiwipiepy import Kiwi
from langchain.retrievers.document_compressors import LLMChainExtractor
from pprint import pprint
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from prompt import Prompt
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI



load_dotenv()

import re
def get_parsed_criminal_law() -> list[dict["title": str, "content": str]]:
    file_name = os.getenv("FILE_NAME")
    data = get_data_from_file(file_name)
    
    # 정규식을 이용하여 "장"을 기준으로 파싱
    pattern = r"(제\d+장)\s([^\n]+)([\s\S]+?)(?=제\d+장|\Z)"
    matches = re.findall(pattern, data)

    result_document_list = []
    for match in matches:
        # 첫 번째와 두 번째 요소를 합쳐서 title로 저장
        title = match[0] + ' ' + match[1]
        article_list = [" "]
        parsed_match = match[2].split('\n')
        for i in range(2, len(parsed_match)):
            raw = parsed_match[i]
            if raw == '' or raw == ' ':
                continue
            if not re.match(r"제\d+조", raw):
                article_list[-1] += raw
            else:
                article_list.append(raw)
        for i in range(1, len(article_list)):
            temp_document = Document(
                page_content=title + '\n' +article_list[i],
                metadata={"title": title}
            )
            result_document_list.append(temp_document)

    return result_document_list  # 파싱된 데이터를 리스트로 반환


def get_embedding() -> OpenAIEmbeddings:
    model = os.getenv("EMBEDDING_MODEL")
    dimensions = os.getenv("EMBEDDING_DIMENSIONS")
    return OpenAIEmbeddings(model=model, dimensions=dimensions)


def create_hybrid_retrieval() -> EnsembleRetriever:
    print("🚀 create_hybrid_retrieval 생성 중...", end="")
    documents: Document = get_parsed_criminal_law()
    # compressor = LLMChainExtractor.from_llm(get_OPENAI_llm(model="gpt-4o-mini", temperature=0.7))

    # dense_vector_retriever 생성
    dense_vector_retriever: VectorStoreRetriever = create_dense_vector_retrieval(documents)
    # compressed_dense_vector_retriever = ContextualCompressionRetriever(
    #     base_compressor=compressor, base_retriever=dense_vector_retriever
    # )

    # sparse_vector_retriever 생성
    sparse_vector_retriever: BM25Retriever = create_sparse_vector_retriever(documents)
    # compressed_sparse_vector_retriever = ContextualCompressionRetriever(
    #     base_compressor=compressor, base_retriever=sparse_vector_retriever
    # )

    # EnsembleRetriever 생성
    retriever: EnsembleRetriever = EnsembleRetriever(
        retrievers = [dense_vector_retriever, sparse_vector_retriever],
        weights = [0.7, 0.3],
        # TODO: 현재 default, 하이퍼 파라미터 튜닝 필요
        c = 60
    )
    print("✅ create_hybrid_retrieval 생성 완료!")

    return retriever


# 클라우드 pinecone에 인덱스 생성
def create_pinecone_index():
    index_name = os.getenv("INDEX_NAME")
    dimension = int(os.getenv("EMBEDDING_DIMENSIONS"))
    pinecone = Pinecone()
    pinecone.create_index(
      name = index_name,
      dimension= dimension,
      # TODO: cosine vs dotproduct
      metric='cosine',
      spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

# 파일에서 데이터를 읽어오는 함수
def get_data_from_file(file_name: str) -> str:
  with open(file_name, 'r') as f:
    data = f.read()
  return data

# 벡터를 인덱스에 추가하는 함수
def add_vector_to_index(vectorStore: PineconeVectorStore, documents: list[Document]) -> None:
    vectorStore.add_documents(documents)
    return

# sparse_vector_retriever 생성
def create_sparse_vector_retriever(docs: Document) -> BM25Retriever:
    print("\n","L", end="")
    print("🚀 sparse_vector_retrieval 생성 중...", end="")
    kiwi = Kiwi()
    def kiwi_tokenize(text):
        return [token.form for token in kiwi.tokenize(text)]

    kiwi_bm25 = BM25Retriever.from_documents(docs, preprocess_func=kiwi_tokenize)
    print("✅ sparse_vector_retrieval 생성 완료!")
    return kiwi_bm25

# dense_vector_retriever 생성
def create_dense_vector_retrieval(documents: Document) -> VectorStoreRetriever:
    print("\n","L", end="")
    print("🚀 dense_vector_retrieval생성 중...", end="")
    pinecone = Pinecone()
    index_name = os.getenv("INDEX_NAME")
    embedding = get_embedding()

    # 만약 pinecone에 인덱스가 없다면 생성
    if index_name not in pinecone.list_indexes().names():
        print("\n","L", end="")
        print("🚀 dense_vector_retrieval 생성 중...", end="")
        create_pinecone_index()
        print("✅ pinecone index 생성 완료!")

    # PineconeVectorStore 불러오기
    index = pinecone.Index(index_name)
    vector_store = PineconeVectorStore(index=index, embedding=embedding)

    # vector_store에 데이터가 없으면 데이터를 추가
    if (index.describe_index_stats()["total_vector_count"] == 0):
        print("\n","L", end="")
        print("🚀 pinecone index에 데이터 추가중 .", end="")
        add_vector_to_index(vector_store, documents)
        print("✅ pinecone index에 데이터 추가 완료!")

    # VectorStoreRetriever 생성
    retriever = vector_store.as_retriever(
        # TODO: mmr vs similarity search 확인요함
        search_type="similarity",
        search_kwargs={"k": 3},
    )
    print("✅ dense_vector_retrieval생성 완료!")

    return retriever




def get_OPENAI_llm(
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
    ) -> BaseChatModel:
    """Returns OpenAI chat models.
    Available OpenAI models: https://platform.openai.com/docs/models/gpt-3-5-turbo
    """
    llm = ChatOpenAI(
        model=model,
        temperature=temperature)      
    return llm



def search_web_reference(search_trem: str) -> str:
    search = GoogleSerperAPIWrapper(hl= "ko", gl= "kr", k=25)
    result = search.run(search_trem)
    return result


def create_chain():
  
  retriever = create_hybrid_retrieval()


  llm = get_OPENAI_llm("gpt-4o-mini", 0.7)
  prompt_instance = Prompt()
  retrieval_qa_chat_prompt = prompt_instance.get_criminal_law_qa_prompt()
  
  combine_docs_chain = create_stuff_documents_chain(
      llm, retrieval_qa_chat_prompt
  )


  retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
  return retrieval_chain


def solve_and_check_answer(item, retrieval_chain):
  result = retrieval_chain.invoke({
    "input": item['question'] + '\n' + item['A'] + '\n' + item['B'] + '\n' + item['C'] + '\n' + item['D'],
    "question": item['question'], 
    "A": item['A'],
    "B": item['B'],
    "C": item['C'],
    "D": item['D'],
    })
  # pprint(result)
  parsed_result = output_parser(result)

  answer_letter = parsed_result['answer_letter']
  final_answer = parsed_result['final_answer']
  thinking_process = parsed_result['thinking_process']
  # Extract the answer letter (A, B, C, or D) from the final answer
  answer_number: int = map_answer_to_number(answer_letter)
  is_correct: bool = answer_number == item['answer']
  
  return {
    "final_answer": final_answer,
    "thinking_process": thinking_process,
    "original_answer": item['answer'],
    "generated_answer": answer_number,
    "is_correct": is_correct
  }

def output_parser(output):
  answer = output['answer']
  thinking_process = output['answer'].split('final_answer:')[0].strip()
  final_answer = output['answer'].split('final_answer:')[1].strip()
  # Extract the answer letter (A, B, C, or D) from the final answer
  
  answer_letter = final_answer.split(' ')[-1][1]
  # if answer_letter not in ['A', 'B', 'C', 'D']:
  #   answer_letter = final_answer.split(' ')[-1][0]
  
  return {
      "thinking_process": thinking_process,
      "final_answer": final_answer,
      "answer_letter": answer_letter
  }


def map_answer_to_number(answer_letter) -> int:
  answer_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
  return answer_mapping.get(answer_letter, None)