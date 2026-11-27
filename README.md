# Chicken Necropsy — 닭 장기 전용 YOLO 로컬 학습 및 추론

## 1. 데이터 준비
[BioVision Chicken v1](https://universe.roboflow.com/na-u0cmg/biovision_chicken/dataset/1) (na, CC BY 4.0, 1022 generated images).
Download Dataset → YOLOv8 format → ZIP 다운로드 → 압축 해제하여 저장소의 dataset/ 폴더에 data.yaml, train/images, train/labels, valid/images, valid/labels 등이 오도록 배치. 사이트 로그인/다운로드 권한이 필요할 수 있습니다.
또는 선택적으로 API로 데이터만 다운로드:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install roboflow
python download_dataset.py
```
API 키는 데이터 다운로드에만 쓰이며 학습 및 추론에는 사용하지 않습니다. 다운로드 접근 실패 시 웹 ZIP 방식 사용.

## 2. 로컬 학습
```bash
bash train.sh
```
YOLO11n 사전 학습 기반 전이학습, Apple GPU MPS 자동 선택, 60 epochs, batch 8, 640px. 최초 기반 모델 다운로드 및 라이브러리 설치는 인터넷 필요. 학습 결과 models/best.pt, 상세 지표 runs/organ*.
다른 데이터 위치: bash train.sh --data /path/to/data.yaml
GPU 오류: bash train.sh --device cpu
메모리 부족: bash train.sh --batch 4
짧은 테스트: bash train.sh --epochs 1 (정확도 평가용 아님).

## 3. 웹앱
```bash
bash run.sh
```
http://127.0.0.1:8501. 여러 사진 업로드 → 전체 사진 자동 분석. 장기별 오버레이, 신뢰도 필터, 원본 비교, JPG/JSON ZIP 다운로드. API 키 및 클라우드 추론 없음. models/best.pt가 없으면 분석 비활성화.

## 제한
실제 데이터 다운로드, 학습, 맥북 실행은 아직 검증하지 못했습니다. 공개 데이터의 증강 이미지와 동일 개체 중복 가능성을 고려하여 독립 부검 사진에서 성능을 평가해야 합니다. 질병 진단용 아님. Ultralytics AGPL-3.0 또는 기업 라이선스 조건 및 데이터 출처 표시 확인 필요. 사진/가중치/데이터는 Git에 올리지 않도록 제외했습니다.
