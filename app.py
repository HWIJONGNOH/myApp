import streamlit as st
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import numpy as np

# --- 1. 기본 설정 및 모델 로드 ---
st.set_page_config(page_title="YOLOv8 Object Detection", layout="centered")
st.title("🔍 YOLOv8 실시간 이미지 탐지 서버")
st.write("학습된 모델을 활용해 이미지 내 물체를 탐지합니다.")

# 모델 경로는 고정되어 있다고 가정합니다.
MODEL_PATH = Path("runs/detect/train/weights/best.pt")

@st.cache_resource # 모델을 한 번만 로드하도록 캐싱 (속도 향상)
def load_yolo_model(path):
    if not path.exists():
        return None
    return YOLO(path)

model = load_yolo_model(MODEL_PATH)

if model is None:
    st.error(f"❌ 모델 파일을 찾지 못했습니다: {MODEL_PATH.resolve()}")
    st.stop() # 모델이 없으면 프로세스 중단

# --- 2. 이미지 업로드 UI ---
uploaded_file = st.file_uploader("탐지할 이미지를 업로드하세요...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # 업로드된 파일을 PIL 이미지로 변환
    input_image = Image.open(uploaded_file)
    
    # 레이아웃을 반으로 나누어 좌측은 원본, 우측은 결과 출력
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("원본 이미지")
        st.image(input_image, use_container_width=True)
        
    with col2:
        st.subheader("탐지 결과")
        
        # 예측 진행 (Streamlit에서는 굳이 파일로 매번 저장하지 않고 메모리 상에서 처리하는 것이 효율적입니다)
        # 하지만 기존 코드의 save=True 설정을 유지하여 하위 폴더에 기록도 남기도록 세팅했습니다.
        with st.spinner("분석 중..."):
            results = model.predict(
                source=input_image, # PIL 이미지를 바로 넣을 수 있습니다.
                save=True,
                conf=0.05,
                project="outputs/new_image_prediction",
                name="result",
                exist_ok=True,
                show_boxes=True,
                show_labels=True,
                show_conf=True,
                line_width=2
            )
            
            # 예측된 결과 플롯 이미지 가져오기 (메모리에서 바로 읽어오기)
            # results[0].plot()은 BGR(OpenCV 형태) 배열을 반환하므로 RGB로 변환해줍니다.
            res_plotted = results[0].plot()[:, :, ::-1] 
            st.image(res_plotted, use_container_width=True)

    # --- 3. 하단에 탐지 텍스트 정보 출력 ---
    st.markdown("---")
    st.subheader("📊 세부 탐지 정보")
    
    r = results[0]
    st.write(f"**탐지된 박스 수:** {len(r.boxes)}개")
    
    if len(r.boxes) > 0:
        # 결과를 깔끔하게 표(Table) 형태로 보여주기 위한 데이터 정렬
        box_data = []
        for i, box in enumerate(r.boxes):
            class_id = int(box.cls.item())
            class_name = model.names[class_id]
            conf = float(box.conf.item())
            xyxy = box.xyxy.cpu().numpy().ravel().tolist()
            
            box_data.append({
                "No": i,
                "Class": class_name,
                "Confidence": f"{conf:.4f}",
                "BBox (xyxy)": [round(coord, 1) for coord in xyxy]
            })
        
        # 스트림릿 테이블로 출력
        st.table(box_data)
    else:
        st.info("탐지된 물체가 없습니다. 신뢰도(conf) 기준을 조절해 보세요.")
