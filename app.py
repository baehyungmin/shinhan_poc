import streamlit as st
import asyncio
import os
# from dotenv import load_dotenv

# from mcp.client.stdio import stdio_client # ⬅️ mcp 관련 로직 제거
# from mcp import ClientSession, StdioServerParameters # ⬅️ mcp 관련 로직 제거
# from langchain_mcp_adapters.tools import load_mcp_tools # ⬅️ mcp 관련 로직 제거
# from langgraph.prebuilt import create_react_agent # ⬅️ mcp/langgraph 관련 로직 제거

from google import genai
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ⬅️ embedding model 추가
from langchain_openai import OpenAIEmbeddings # ⬅️ Langchain의 최신 OpenAI 임베딩 모듈
from pymilvus import MilvusClient # ⬅️ Milvus client 추가

from PIL import Image
from pathlib import Path

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
ZILLIZ_URI = st.secrets["ZILLIZ_URI"]
ZILLIZ_TOKEN = st.secrets["ZILLIZ_TOKEN"]

EMBEDDING_MODEL='text-embedding-3-small'
EMBEDDING_DIMENSION =1536
COLLECTION_NAME='shinahn_collection'
MODEL_NAME = 'gemini-2.5-flash'

milvus_client = MilvusClient(uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
gemini_client  = genai.Client()

# 환경변수
ASSETS = Path("assets")
# GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
# # ⬅️ OpenAI 및 Milvus 관련 환경 변수 추가 (st.secrets에 저장되어 있다고 가정)
# OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"] # OpenAI API Key
# MILVUS_URI = st.secrets["MILVUS_URI"]
# MILVUS_TOKEN = st.secrets["MILVUS_TOKEN"]
# COLLECTION_NAME = st.secrets["COLLECTION_NAME"]

# # 모델 및 시스템 설정
# MODEL_NAME = 'gemini-2.5-flash'
SYSTEM_PROMPT = "당신은 친절한 마케팅 상담사입니다. 가맹점명을 받아 해당 가맹점의 방문 고객 현황을 분석하고, 분석 결과를 바탕으로 적절한 마케팅 방법과 채널, 마케팅 메시지를 추천합니다. 결과는 짧고 간결하게, 분석 결과에는 가능한 표를 사용하여 알아보기 쉽게 설명해주세요."
GREETING = "마케팅이 필요한 가맹점을 알려주세요 \n(조회가능 예시: 동대*, 유유*, 똥파*, 본죽*, 본*, 원조*, 희망*, 혁이*, H커*, 케키*)"

# ---------------------------------------------------------------------------------
# 🔍 Embedding 및 Retrieval 함수
# ---------------------------------------------------------------------------------

embedding_model = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        model=EMBEDDING_MODEL,
        # 'dimensions' 파라미터를 사용하여 벡터 길이를 1536으로 고정합니다.
        dimensions=EMBEDDING_DIMENSION
)

@st.cache_resource
def embed_query(query: str, embedding_model):
    """사용자 쿼리를 벡터로 임베딩합니다."""
    # Langchain Embeddings 인터페이스 사용: embed_query
    return embedding_model.embed_query(query)

def retrieve_from_milvus(query_vector: list):
    """Milvus Cloud에서 가장 유사한 top_k=1의 결과를 가져옵니다."""
    try:
        search_vectors = [query_vector]
        output_fields = ["text", "description"]

        # Milvus 검색
        search_result = milvus_client.search(
            collection_name=COLLECTION_NAME,
            data=search_vectors,      # 검색할 쿼리 벡터
            limit=1,                   # 상위 1개의 결과만 가져옴 (요청에 따라 top_k=1)
            output_fields=output_fields # ⬅️ 검색 결과에 포함할 필드 지정
        )

        # 결과 파싱: 요청된 형식에 따라 가장 첫 번째 엔티티의 'description' 필드 반환
        if search_result and search_result[0]:
            # result = search_result[0][0]['entity']['description']
            # description 필드가 검색 결과의 RAG 컨텍스트 데이터라고 가정
            result = search_result[0][0]['entity']['description']
            return result
        return None

    except Exception as e:
        st.error(f"Milvus 검색 중 오류 발생: {e}")
        return None


@st.cache_data
def load_image(name: str):
    return Image.open(ASSETS / name)

st.set_page_config(page_title="2025년 빅콘테스트 AI데이터 활용분야 - 맛집을 수호하는 AI비밀상담사")

def clear_chat_history():
    # SYSTEM_PROMPT 상수로 변경
    st.session_state.messages = [SystemMessage(content=SYSTEM_PROMPT), AIMessage(content=GREETING)]

# 사이드바
with st.sidebar:
    st.image(load_image("shc_ci_basic_00.png"), width='stretch')
    st.markdown("<p style='text-align: center;'>2025 Big Contest</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>AI DATA 활용분야</p>", unsafe_allow_html=True)
    st.write("")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.button('Clear Chat History', on_click=clear_chat_history)

# 헤더
st.title("신한카드 소상공인 🔑 비밀상담소")
st.subheader("#우리동네 #숨은맛집 #소상공인 #마케팅 #전략 .. 🤤")
st.image(load_image("image_gen3.png"), width='stretch', caption="🌀 머리아픈 마케팅 📊 어떻게 하면 좋을까?")
st.write("")

# 메시지 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        AIMessage(content=GREETING)
    ]

# 초기 메시지 화면 표시
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

def render_chat_message(role: str, content: str):
    with st.chat_message(role):
        st.markdown(content.replace("<br>", " \n"))



# 사용자 입력 처리 
async def generate_answer_with_description_rag(gemini_client, user_query: list):
    """
    검색된 컨텍스트(문서 내용 + description 메타데이터)를 기반으로 Gemini에게 답변을 요청합니다.
    """
    query_vector = embed_query(query, embedding_model)
    full_context = retrieve_from_milvus(query_vector)


    # 프롬프트 구성
    prompt = f"""
      아래의 데이터는 JSON 형식의 데이터로 특정 가맹점의 정보와 그 가맹점의 최근 24 개월간의 월별 이용 정보와 월별 이용 고객 정보 데이터이며 시간순으로 정렬이 되어 있어.
      - 값이 -999999.99 인 경우 정보가 존재하지 않는 경우임
      - '가맹점 운영 개월수 구간': 0% 에 가까울 수록 상위임(운영개월 수가 상위 임)
      - '매출금액 구간': 0% 에 가까울 수록 상위임(매출 금액이 상위 임)
      - '매출건수 구간': 0% 에 가까울 수록 상위임(매출 건수가 상위 임)
      - '유니크 고객 수 구간': 0% 에 가까울 수록 상위임(unique 고객 수가 상위임)
      - '객단가 구간': 0% 에 가까울 수록 상위임(객단가가 상위 임)
      - '취소율 구간': 1 구간에 가까울 수록 취소율이 낮음
      - '동일 업종 매출금액 비율': 동일 업종 매출 금액 평균 대비 해당 가맹점의 매출 금액 비율이며 평균과 동일할 경우 100 이야.
      - '동일 업종 매출건수 비율': 동일 업종 매출 건수 평균 대비 해당 가맹점의 매출 건수 비율이며 평균과 동일할 경우 100 이야.
      - '동일 업종 내 매출 순위 비율': ('업종 내 순위'/'업종 내 전체 가맹점'* 100) 을 계산한 값으로 0 에 가까울 수록 상위에 랭킹되는거야.
      - '동일 상권 내 매출 순위 비율': ('상권 내 순위'/'상권 내 전체 가맹점'* 100) 을 계산한 값으로 0 에 가까울 수록 상위에 랭킹되는거야.

      아래의 data 를 분석해서 해당 가맹점의 매출 전략을 제안해줘.

    {full_context}

    ---

    QUERY:
    {user_query}
    """

    response = gemini_client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={
            "temperature": 0.1 # 사실 기반 답변을 위해 낮은 온도를 사용
        }
    )

    return response.text


# 사용자 입력 창
if query := st.chat_input("가맹점 이름을 입력하세요"):
    # 사용자 메시지 추가
    st.session_state.messages.append(HumanMessage(content=query))
    render_chat_message("user", query)

    with st.spinner("Thinking..."):
        try:
            # RAG 로직 실행
            # process_user_input_rag 함수는 내부에서 동기 함수를 호출하므로, asyncio.run을 제거하고 직접 호출합니다.
            # 하지만 Streamlit의 UI 업데이트를 위해 process_user_input_rag를 async로 유지하고 asyncio.run을 사용합니다.
            # RAG 함수는 실제 비동기 I/O를 수행하지 않으므로, 사실상 `await process_user_input_rag(query)`로 변경하는 것이 더 자연스럽지만
            # 기존 코드 구조를 유지하기 위해 asyncio.run을 사용합니다.

            # 기존 코드의 구조를 유지하며, RAG 함수는 async로 정의합니다.
            # st.chat_input 콜백은 동기 컨텍스트이므로 asyncio.run을 사용합니다.
            reply = asyncio.run(generate_answer_with_description_rag(gemini_client=gemini_client,user_query=query))

            st.session_state.messages.append(AIMessage(content=reply))
            render_chat_message("assistant", reply)

        except Exception as e:
            # 단일 예외 처리
            error_msg = f"오류가 발생했습니다: {e!r}"
            st.session_state.messages.append(AIMessage(content=error_msg))
            render_chat_message("assistant", error_msg)