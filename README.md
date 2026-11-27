# Chicken Necropsy — 가금류 부검 장기 인식

사전 학습된 BioVision_Chicken 모델을 이용한 다중 사진 분석 웹앱. 별도 모델 학습은 하지 않습니다.

## macOS 실행 (Python 3.11 또는 3.12 권장)

```bash
git clone https://github.com/tajogood/chicken_bugum.git
cd chicken_bugum
bash run.sh
```

브라우저에서 http://127.0.0.1:8501 접속. Roboflow 계정의 API Key를 사이드바에 입력하거나 .env.example을 .env로 복사하여 ROBOFLOW_API_KEY를 설정하세요. 키를 Git에 커밋하지 마세요.

사진 여러 장 선택 → 외부 전송 동의 → 전체 사진 자동 분석. 신뢰도 조절, 장기 필터, 원본 비교, JPG 및 JSON ZIP 저장을 지원합니다.

## 모델 및 제한

- [BioVision_Chicken](https://universe.roboflow.com/na-u0cmg/biovision_chicken), model ID `biovision_chicken/1`, Roboflow 3.0 Object Detection (Fast), 프로젝트 표기 CC BY 4.0, by na (2024).
- 공식 serverless API를 사용합니다. 웹앱과 결과 렌더링은 로컬이지만 사진은 Roboflow 서버로 전송됩니다. 인터넷 및 유효한 API 키가 필요하고 계정 사용량 제한/비용이 적용될 수 있습니다. 오프라인 모델 가중치는 포함하지 않았습니다.
- 9개 장기: 간, 심장, 폐, 신장, 비장, 담낭, 근위, 소장, 대장. 난소, 구강, 기관, 지방, 병변 등은 학습 클래스가 아닙니다.
- 모델 출력은 사각형 검출이며 픽셀 단위 분할이 아닙니다. 신뢰도는 확진 확률이 아닙니다. 모델 페이지의 평가 성능은 실제 부검 사진에서 보장되지 않습니다.
- 사진은 EXIF 방향을 보정하고 최대 2048px로 축소하여 분석합니다. API 요청 기준 신뢰도는 0.10, 화면 기준은 변경 가능합니다. 0.10 미만의 검출은 반환되지 않을 수 있습니다.
- 사진은 GitHub에 저장하지 않으며 앱에서 임시 파일은 자동 삭제합니다. 외부 서비스의 보관 정책 및 이용권한은 별도로 확인하세요.
- 연구용 PoC이며 질병 진단 기능은 없습니다.

## 수동 실행

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```
