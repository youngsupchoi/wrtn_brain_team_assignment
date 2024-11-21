from re import search
from typing import final
from weakref import ref
from langchain_core.documents import Document
from langsmith import evaluate
from prompt import Prompt
from tools import create_hybrid_retrieval, get_OPENAI_llm, search_web_reference
from langchain_core.prompts.chat import ChatPromptTemplate
from pydantic import BaseModel
from pprint import pprint
import random

class targetBooleanResponse(BaseModel):
  thinking_process: str
  target: bool

class searchTermResponse(BaseModel):
  search_term: str

class evaluateTruthfulnessResponse(BaseModel):
  thinking_process: str
  final_answer: bool

class competeChoicesResponse(BaseModel):
  thinking_process: str
  final_answer: int



class Agent():
  def __init__(self) -> None:
    self.prompt_instance = Prompt()
    self.hybrid_retrieval = create_hybrid_retrieval()

  # 주어진 문제의 목표가 참인 것을 찾는 것인지 거짓인 것을 찾는 것인지 판단하는 agent
  def find_target_bool_type(self, question: str) -> bool:
    
    prompt: ChatPromptTemplate = self.prompt_instance.get_find_target_prompt()
    llm = get_OPENAI_llm("gpt-4o-mini", 0.2).with_structured_output(targetBooleanResponse)

    chain = prompt | llm
    result = chain.invoke({
      "question": question
    })
    
    return result.target
  
  def creaet_search_term(self, question: str, choice: str) -> str:
    prompt: ChatPromptTemplate = self.prompt_instance.get_creaet_search_termrompt()
    llm = get_OPENAI_llm("gpt-4o-mini", 0.7).with_structured_output(searchTermResponse)
    chain = prompt | llm
    result = chain.invoke({
      "question": question,
      "choice": choice
    })
    return result.search_term

  
  # 문제와 하나의 선지가 주어졌을 때, 해당 선지가 참인지 거짓인지 판단하는 함수
  def evaluate_truthfulness_for_one(self, question: str, choice: str) -> bool:
    prompt: ChatPromptTemplate = self.prompt_instance.get_evaluate_truthfulness_prompt()
    llm = get_OPENAI_llm("gpt-4o-mini", 0.7).with_structured_output(evaluateTruthfulnessResponse)
    
    web_reference: str = self.get_web_reference(question, choice)
    law_reference: str = self.get_law_reference(question, choice)

    chain = prompt | llm
    result = chain.invoke({
      "question": question,
      "choice": choice,
      "law_reference": law_reference,
      "web_reference": web_reference
    })
    return {
      "choice": choice,
      "question": question,
      "final_answer": result.final_answer,
      "thinking_process": result.thinking_process,
      "web_reference": web_reference,
      "law_reference": law_reference
    }

  # 질문과 하나의 선지를 기반으로 하이브리드 벡터서치를 진행하여 관련된 law reference를 가져오는 함수
  def get_law_reference(self, question: str, choice: str) -> str:
    retriever = self.hybrid_retrieval
    result: list[Document] = retriever.invoke(question + '\n' + choice)
    reference = ""

    for i in range(0, len(result)):
      reference += result[i].page_content
      reference += '\n'
    # TODO: compressor추가
    return reference

  # 질문과 하나의 선지를 기반으로 하이브리드 벡터서치를 진행하여 관련된 web reference를 가져오는 함수
  def get_web_reference(self, question:str, choice: str) -> str:
    search_term: str = self.creaet_search_term(question, choice)
    web_reference: str = search_web_reference(search_term)

    # TODO: compressor추가
    return web_reference
  
  # target bool에 부학하는 선지가 2개 이상일때 경쟁을 통해 최종 답을 도출하는 함수
  def compete_choices(self, choices: list[dict], question: str) -> int:
    merged_thinking_process_and_choice = ""

    for choice in choices:
      merged_thinking_process_and_choice += choice["thinking_process"]
      merged_thinking_process_and_choice += '\n'
      merged_thinking_process_and_choice += str(choice["answer_number"]) + ". " + choice["choice"] + '\n'
      merged_thinking_process_and_choice += '\n'
      

    prompt: ChatPromptTemplate = self.prompt_instance.get_compete_choices_prompt()
    llm = get_OPENAI_llm("gpt-4o-mini", 0.7).with_structured_output(competeChoicesResponse)
    
    chain = prompt | llm
    result = chain.invoke({
      "question": question,
      "merged_thinking_process_and_choice": merged_thinking_process_and_choice,
    })
    
    
    return result.final_answer
  # 여러개의 선지를 각각 평가하고 이를 종합하여 최종 답과 정답 여부를 반환
  def retrieve_and_summarize(self, item: dict):
    retrieved_final_answer = -1

    # TODO: 병렬 처리 요함
    # 각 선지에 대해 참인지 거짓인지 판단
    A_result = self.evaluate_truthfulness_for_one(item["question"], item["A"])
    B_result = self.evaluate_truthfulness_for_one(item["question"], item["B"])
    C_result = self.evaluate_truthfulness_for_one(item["question"], item["C"])
    D_result = self.evaluate_truthfulness_for_one(item["question"], item["D"])
    # print("A_result", A_result)
    # print("B_result", B_result)
    # print("C_result", C_result)
    # print("D_result", D_result)

    # 현재 question의 옳은 것을 찾으려는지 틀린 것을 찾으려는지 판단
    target_bool: bool = self.find_target_bool_type(item)
    # print("target_bool", target_bool)

    A_result['answer_number'] = 1
    B_result['answer_number'] = 2
    C_result['answer_number'] = 3
    D_result['answer_number'] = 4
    
    result = []
    if A_result["final_answer"] == target_bool:
      
      result.append(A_result)
    if B_result["final_answer"] == target_bool:
      
      result.append(B_result)
    if C_result["final_answer"] == target_bool:
      
      result.append(C_result)
    if D_result["final_answer"] == target_bool:
    
      result.append(D_result)
    # 바람직한 상황, 하나의 답이 결정됨
    if len(result) == 1:
      retrieved_final_answer: int = result[0]["answer_number"]
    # 바람직하지 못한 상황, 답으로 나온 결과가 없음
    if len(result) == 0:
      # 모든 항목을 넣고 2개 이상일때와 같은 로직으로 처리
      result = [A_result, B_result, C_result, D_result]
    # 바람직하지 못한 상황, 2개 이상의 답이 나옴, 경쟁을 통해 최종 답을 도출
    if len(result) > 1:
      question = item["question"]
      retrieved_final_answer = self.compete_choices(result, question)
    
    # 정답과 비교하여 정답 여부 판단
    if retrieved_final_answer == item["answer"]:
      return_result = {
      "selected_choices": result,
      "selected_choices_count": len(result),
      "generated_answer": retrieved_final_answer,
      "is_correct": True,
      "original_answer": item["answer"],
      }
      return return_result
    else:
      return_result = {
        "selected_choices": result,
        "selected_choices_count": len(result),
        "generated_answer": retrieved_final_answer,
        "is_correct": False,
        "original_answer": item["answer"],
      }
    
      return return_result



# from datasets import load_dataset
# ds = load_dataset("HAERAE-HUB/KMMLU", "Criminal-Law")

# test = ds['test']

# item = test[2]
# pprint(item)

# train = ds['train']
# dev = ds['dev']
# test = ds['test']

# agent = Agent()
# # print(agent.creaet_search_term(question= "공소장 변경에 대한 설명으로 옳지 않은 것은?", choice= "법원은 검사가 공소장 변경을 신청한 경우 피고인이나 변호인의 청구가 있는 때에는 피고인으로 하여 금 필요한 방어의 준비를 하게 하기 위해 필요한 기간 공판 절차를 정지하여야 한다."))
# pprint(agent.retrieve_and_summarize(item))