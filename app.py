import os
import io
import json
import zipfile
import hashlib
from pathlib import Path
from collections import Counter

import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps
os.environ.setdefault("YOLO_OFFLINE", "true")
MODEL = str(Path(__file__).resolve().parent / "models" / "best.pt")
NAMES = {
    "gallbladder": "담낭", "gizzard": "근위", "heart": "심장",
    "kidney": "신장", "large_intestine": "대장", "liver": "간",
    "lungs": "폐", "small_intestine": "소장", "spleen": "비장", "ovary": "난소", "ovarian_follicle": "난포", "trachea": "기관", "tongue": "혀", "abdominal_fat": "복부 지방",
}
COLORS = ["#36d399", "#60a5fa", "#fbbf24", "#f472b6", "#a78bfa",
          "#fb923c", "#22d3ee", "#f87171", "#a3e635"]

def name_ko(label):
    return NAMES.get(label.lower().replace(" ", "_"), label)

def color(label):
    key = label.lower().replace(" ", "_")
    return COLORS[list(NAMES).index(key) % len(COLORS)] if key in NAMES else "#ffffff"

def jpeg(image):
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=94)
    return out.getvalue()

def read_image(data):
    with Image.open(io.BytesIO(data)) as im:
        image = ImageOps.exif_transpose(im).convert("RGB")
    image.thumbnail((2048, 2048))
    return image

def font_for(size):
    for path in ["/System/Library/Fonts/AppleSDGothicNeo.ttc",
                 "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
                 "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size), True
            except OSError:
                pass
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size), False
    except OSError:
        return ImageFont.load_default(), False

def overlay(image, result, threshold=0.35, selected=None):
    w, h = image.size
    metadata = result.get("image") or {}
    sx = w / (metadata.get("width") or w)
    sy = h / (metadata.get("height") or h)
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font, korean = font_for(max(15, round(w / 55)))
    stroke = max(2, round(w / 350))
    rows = []
    for p in result.get("predictions", []):
        label = str(p["class"])
        confidence = float(p["confidence"])
        if confidence < threshold or (selected is not None and label not in selected):
            continue
        x, y, bw, bh = (float(p[k]) for k in ("x", "y", "width", "height"))
        x1 = max(0, min(w-1, round((x-bw/2)*sx)))
        y1 = max(0, min(h-1, round((y-bh/2)*sy)))
        x2 = max(0, min(w-1, round((x+bw/2)*sx)))
        y2 = max(0, min(h-1, round((y+bh/2)*sy)))
        if x2 <= x1 or y2 <= y1:
            continue
        c = color(label)
        draw.rectangle((x1,y1,x2,y2), outline=c, width=stroke)
        caption = f"{name_ko(label) if korean else label} {confidence:.0%}"
        bounds = draw.textbbox((0,0), caption, font=font)
        tw, th = bounds[2]-bounds[0]+10, bounds[3]-bounds[1]+10
        tx = max(0, min(x1, w-tw))
        ty = max(0, y1-th)
        draw.rectangle((tx,ty,min(w-1,tx+tw),min(h-1,ty+th)), fill=c)
        draw.text((tx+5,ty+5-bounds[1]),caption,font=font,fill="#101820")
        rows.append({"장기": name_ko(label), "Class": label, "신뢰도": round(confidence,4),
                     "bbox_xyxy": [x1,y1,x2,y2]})
    return canvas, rows

@st.cache_resource(show_spinner=False)
def load_model(weight_mtime):
    from ultralytics import YOLO
    return YOLO(MODEL)

def infer(model, image):
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    result = model.predict(image, conf=0.1, imgsz=640, device=device, verbose=False)[0]
    predictions = []
    for box in result.boxes:
        x,y,w,h = box.xywh[0].cpu().tolist()
        label = result.names[int(box.cls.item())]
        predictions.append(dict(x=x,y=y,width=w,height=h,confidence=float(box.conf.item()),class_=label))
        predictions[-1]["class"] = predictions[-1].pop("class_")
    w,h = image.size
    return {"image":{"width":w,"height":h},"predictions":predictions}

def main():
    st.set_page_config(page_title="가금류 부검 분석", page_icon="🔬", layout="wide")
    st.title("가금류 부검 사진 · 장기 인식")
    st.caption("YOLO11 / 닭 장기 전용 로컬 추론 · 연구용")
    st.warning("장기 인식 실험용입니다. 질병 진단 또는 정상 판정 기능은 아닙니다.")
    with st.sidebar:
        st.header("모델 설정")
        st.write(MODEL)
        st.markdown("[모델 정보](https://universe.roboflow.com/na-u0cmg/biovision_chicken)")
        threshold = st.slider("표시 신뢰도 기준", 0.0, 1.0, 0.35, 0.05)
        compare = st.checkbox("원본 함께 표시", value=True)
        st.caption("학습된 models/best.pt를 로컬에서 사용합니다. API 키 불필요.")
        if not Path(MODEL).is_file():
            st.warning("먼저 bash train.sh를 실행하여 모델을 학습하세요.")
    uploads = st.file_uploader("부검 사진 여러 장 선택 또는 드래그", type=["jpg","jpeg","png","webp"], accept_multiple_files=True)
    if "results" not in st.session_state:
        st.session_state.results = {}
    items = []
    for f in uploads or []:
        data = f.getvalue()
        uid = hashlib.sha256(data).hexdigest()
        items.append((uid, f.name, data))
    if st.button("전체 사진 자동 분석", type="primary", disabled=not items or not Path(MODEL).is_file()):
        try:
            with st.spinner("로컬 모델 로딩 중"):
                client = load_model(Path(MODEL).stat().st_mtime_ns)
        except Exception as exc:
            msg = str(exc)
            st.error("로컬 모델을 불러오지 못했습니다: " + msg[:500])
            st.stop()
        st.session_state.results = {}
        bar = st.progress(0)
        for i,(uid,filename,data) in enumerate(items):
            try:
                image = read_image(data)
                st.session_state.results[uid] = {"image": image, "result": infer(client,image)}
            except Exception as exc:
                # Do not expose API secrets potentially embedded in error messages.
                st.session_state.results[uid] = {"error": str(exc)[:500]}
            bar.progress((i+1)/len(items), text=f"{i+1}/{len(items)} 분석 완료")
    available = sorted({str(p["class"]) for uid,_,_ in items
                        for p in st.session_state.results.get(uid,{}).get("result",{}).get("predictions",[])})
    selected = st.multiselect("표시할 장기", available, default=available, format_func=lambda x: f"{name_ko(x)} / {x}") if available else []
    summaries = []
    archive = io.BytesIO()
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        for i,(uid,filename,data) in enumerate(items):
            st.divider()
            st.subheader(f"{i+1}. {filename}")
            entry = st.session_state.results.get(uid)
            if not entry:
                try:
                    st.image(read_image(data),width=350)
                except Exception:
                    st.error("이미지를 읽을 수 없습니다.")
                continue
            if "error" in entry:
                st.error(entry["error"])
                continue
            image = entry["image"]
            result = entry["result"]
            rendered, rows = overlay(image,result,threshold,set(selected))
            if compare:
                a,b = st.columns(2)
                a.image(image,caption="원본",use_container_width=True)
                b.image(rendered,caption="장기 인식 결과",use_container_width=True)
            else:
                st.image(rendered,use_container_width=True)
            st.write(f"표시된 검출: {len(rows)}개")
            if rows:
                st.dataframe(rows,use_container_width=True,hide_index=True)
            else:
                st.info("현재 기준에서 검출된 장기가 없습니다. 정상이라는 뜻은 아닙니다.")
            out = jpeg(rendered)
            stem = f"{i+1:03d}_{Path(filename).stem}"
            st.download_button("오버레이 JPG 저장",out,f"{stem}_overlay.jpg","image/jpeg",key=f"download_{i}")
            z.writestr(f"{stem}_overlay.jpg",out)
            z.writestr(f"{stem}_predictions.json",json.dumps({"model":MODEL,"raw":result,"displayed":rows},ensure_ascii=False,indent=2))
            summaries.append({"file":filename,"detections":len(rows),"organs":dict(Counter(r["장기"] for r in rows))})
        z.writestr("summary.json",json.dumps(summaries,ensure_ascii=False,indent=2))
    if summaries:
        st.sidebar.metric("분석 성공",len(summaries))
        st.sidebar.metric("표시된 검출 합계",sum(s["detections"] for s in summaries))
        st.download_button("전체 결과 ZIP 다운로드",archive.getvalue(),"chicken_necropsy_results.zip","application/zip")
    if st.sidebar.button("분석 결과 초기화"):
        st.session_state.results = {}
        st.rerun()

if __name__ == "__main__":
    main()
