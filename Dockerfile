# Python 3.12.5 slim 이미지 사용
FROM python:3.12-slim

# 작업 디렉토리 설정
WORKDIR /app

# Poetry 설치
RUN pip install --no-cache-dir poetry

# Poetry 설정 파일 복사
COPY pyproject.toml poetry.lock /app/

# Poetry를 사용해 의존성 설치
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# 소스 코드 및 .env 파일 복사
COPY . /app/

# 실행 파일 설정
CMD ["python", "evluation.py"]