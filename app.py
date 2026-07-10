import streamlit as st
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import numpy as np
import os
# 최신 Google GenAI SDK 임포트
from google import genai
from google.genai import types

# --- 1. 페이지 설정 및 초기화 ---
st.set_page_config(page_title="AI 품질 검사 시스템", layout="wide")
st.title("🛠️ YOLOv8 X Gemini AI 품질 검사 및 자동 해석 시스템")
st.write("이미지를 업로드하면 YOLOv8이 불량을 탐지하고, Gemini가 검사 리포트를 생성합니다.")

# 모델 경로 설정
MODEL_PATH = Path("runs/detect/train/weights/best.pt")

@st.cache_resource
def load_yolo_model(path):
    if not path.exists():
        return None
    return YOLO(path)

model = load_yolo_model(MODEL_PATH)

if model is None:
    st.error(f"❌ 모델 파일을 찾지 못했습니다: {MODEL_PATH.resolve()}")
    st.stop()

# --- 2. Gemini API 클라이언트 설정 ---
# 로컬 테스트 시: .streamlit/secrets.toml 에 GEMINI_API_KEY="내키" 입력하여 사용
# 또는 임시로 os.environ["GEMINI_API_KEY"] = "발급받은키" 로 설정 가능합니다.
try:
    # 2026년 기준 표준 SDK 사용 방식 (클라이언트 초기화시 환경변수 자동 참조)
    ai_client = genai.Client()
except Exception as e:
    ai_client = None

# --- 3. 사이드바 설정 (임계치 조절) ---
st.sidebar.header("⚙️ 검사 설정")
conf_threshold = st.sidebar.slider(
    "불량 탐지 임계치 (Confidence Threshold)",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.05
)

# --- 4. 이미지 업로드 ---
uploaded_file = st.file_uploader("검사할 이미지를 업로드하세요...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    
    # YOLOv8 예측 진행
    with st.spinner("YOLOv8 이미지 분석 중..."):
        results = model.predict(
            source=input_image,
            conf=conf_threshold,
            save=False
        )
    
    r = results[0]
    detected_count = len(r.boxes)
    
    # 판정 결과 배너
    if detected_count > 0:
        st.error(f"🚨 **판정 결과: 불량 (Defective)** - 불량 요소 {detected_count}개 감지됨")
    else:
        st.success("✅ **판정 결과: 정상 (Normal)** - 감지된 불량 요소 없음")
        
    st.markdown("---")
    
    # 이미지 시각화 레이아웃
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📸 원본 이미지")
        st.image(input_image, use_container_width=True)
    with col2:
        st.subheader("🔍 탐지 결과 시각화")
        res_plotted = r.plot()[:, :, ::-1]
        st.image(res_plotted, use_container_width=True)

    # 데이터 정리
    box_data = []
    summary_text = ""
    for i, box in enumerate(r.boxes):
        class_id = int(box.cls.item())
        class_name = model.names[class_id]
        conf = float(box.conf.item())
        box_data.append({
            "번호": i + 1,
            "불량 항목 (Class)": class_name,
            "신뢰도 (Confidence)": f"{conf * 100:.1f}%"
        })
        summary_text += f"- {i+1}번 탐지 물체: {class_name} (신뢰도: {conf * 100:.1f}%)\n"

    if detected_count > 0:
        st.subheader("📊 감지된 불량 상세 정보")
        st.table(box_data)
        
    # --- 5. ✨ Gemini API 결과 해석 기능 추가 ---
    st.markdown("---")
    st.subheader("🤖 Gemini AI 품질 분석 리포트")
    
    if ai_client is None and "GEMINI_API_KEY" not in os.environ:
        st.warning("⚠️ Gemini API 키가 설정되지 않았습니다. 사이드바나 Secrets에 API 키를 등록해주세요.")
    else:
        # 프롬프트 구성
        prompt = f"""
        당신은 제조 공정의 AI 품질 검사 전문가입니다. 
        비전 AI 모델(YOLOv8)이 제품 이미지를 분석한 결과 데이터와 원본 이미지를 기반으로 종합 품질 분석 리포트를 작성해주세요.

        [분석 데이터]
        - 최종 판정: {"🚨 불량 (Defective)" if detected_count > 0 else "✅ 정상 (Normal)"}
        - 탐지된 총 불량 개수: {detected_count}개
        - 상세 리스트:
        {summary_text if detected_count > 0 else "없음"}

        [요청 사항]
        1. 현재 제품 상태에 대한 종합 평가를 요약해 주세요.
        2. 탐지된 불량이 있다면, 해당 불량이 제품 품질이나 후속 공정에 미칠 위험성을 설명해 주세요. (정상일 경우 유지보수 팁 제안)
        3. 작업자가 취해야 할 추천 조치 사항(Action Item)을 전문적이고 친절하게 작성해 주세요.
        
        답변은 깔끔한 마크다운 양식으로 작성해 주세요.
        """
        
        # 분석 버튼 제공 (매번 자동으로 API 호출되어 비용이 낭비되는 것을 방지)
        if st.button("🪄 Gemini AI 종합 리포트 생성하기"):
            with st.spinner("Gemini가 이미지를 분석하고 리포트를 작성 중입니다..."):
                try:
                    # 이미지와 텍스트 프롬프트를 함께 전달 (Multimodal)
                    response = ai_client.models.generate_content(
                        model='gemini-3.1-flash-lite',
                        contents=[input_image, prompt]
                    )
                    
                    # 결과 출력
                    st.markdown(response.text)
                    
                except Exception as e:
                    st.error(f"Gemini API 호출 중 오류가 발생했습니다: {e}")
