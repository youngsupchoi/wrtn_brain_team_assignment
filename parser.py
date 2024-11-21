import os
import re
from dotenv import load_dotenv
from tools import get_data_from_file
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document


load_dotenv()
import re
def get_parsed_criminal_law() -> list[dict["title": str, "content": str]]:
    file_name = os.getenv("FILE_NAME")
    data = get_data_from_file(file_name)
    
    # 정규식을 이용하여 "장"을 기준으로 파싱
    pattern = r"(제\d+장)\s([^\n]+)([\s\S]+?)(?=제\d+장|\Z)"
    matches = re.findall(pattern, data)
    
    # 파싱된 결과를 리스트로 저장
    parsed_data = []
    for match in matches:
        
        chapter_title: str = match[1].strip()  # 장 제목
        chapter_data: str = match[2].strip()   # 장 내용
        document: Document = Document(
            page_content=chapter_title + '\n' + chapter_data,
            metadata={"title": chapter_title}
        )
        parsed_data.append(document)
    
    return parsed_data  # 파싱된 데이터를 리스트로 반환