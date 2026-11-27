# Chicken Necropsy — 로컬 장기 인식

BioVision_Chicken 사전 학습 모델을 Roboflow Inference Python 패키지로 맥북에서 직접 실행합니다. 클라우드 추론 호출 및 자동 학습은 없습니다.

## 실행 (Apple Silicon, Python 3.11 권장)

```bash
git clone https://github.com/tajogood/chicken_bugum.git
cd chicken_bugum
bash run.sh
```

기존 사용자: git pull 후 bash run.sh.
http://127.0.0.1:8501 접속 → 사진 여러 장 선택 → 전체 사진 자동 분석.

## 모델
[BioVision_Chicken](https://universe.roboflow.com/na-u0cmg/biovision_chicken) (biovision_chicken/1, 프로젝트 CC BY 4.0).
최초 모델 다운로드/인증에는 인터넷 및 Roboflow 키가 필요할 수 있습니다. 키는 사이드바 또는 .env에 설정합니다. 추론은 get_model().infer()로 로컬에서 수행하며 사진을 hosted inference API에 전송하는 코드가 없습니다. 모델 다운로드 가능 여부 및 macOS 실기기 호환성은 아직 검증하지 못했습니다. 캐시 이후 완전 오프라인 인증 동작도 보장하지 않습니다. 모델 로딩 실패 시 오류를 표시하며 클라우드로 전환하지 않습니다.

9개 장기 검출, 신뢰도 조절, 원본 비교, JPG/JSON ZIP 저장. 사각형 검출이며 질병 진단용이 아닙니다. 원본 모델 및 라이브러리 라이선스 확인 필요.
