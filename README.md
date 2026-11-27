# Chicken Necropsy — API 키 없는 로컬 장기 인식 실험

Roboflow BioVision_Chicken 모델 다운로드 권한 오류로 기본 모델을 [Grounding DINO Tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny)로 교체했습니다. 닭 장기 특화 모델이 아닌 범용 zero-shot 검출 모델입니다. 성능은 검증되지 않았으며 질병 진단에 사용하지 마세요.

## macOS (Python 3.11 권장)
기존 저장소에서:
```bash
git pull
bash run.sh
```
http://127.0.0.1:8501 접속. 여러 사진을 선택하고 전체 사진 자동 분석을 누릅니다.
최초 실행 시 Hugging Face에서 공개 모델 가중치를 다운로드합니다. API 키 불필요. CPU 로컬 추론이며 사진은 외부 API에 전송하지 않습니다. 가중치는 로컬 캐시에 보관됩니다. 오프라인 캐시 사용 시 HF_HUB_OFFLINE=1 bash run.sh (이미 필요한 파일이 다운로드되어 있어야 함).
원본 비교, 신뢰도 및 장기 필터, JPG 및 JSON ZIP 저장 지원. 장기 위치는 사각형으로 표시. 실제 부검 사진 및 M5 환경에서 아직 실행 검증하지 못했습니다.

## 모델 교체 이력
BioVision_Chicken/1 전용 모델은 사용자 환경에서 리소스 접근 오류가 발생하여 제거했습니다. 이전 구현은 Git 이력에 남아 있습니다.
