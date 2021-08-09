# 맥락정보를 가진 혐오(Toxic) 표현 분류 문제
### **코드스테이츠 기업협업프로젝트 With '마인드로직'**

## ***프로젝트 개요***
**프로젝트 문제 정의**
- 혐오 발언의 검출을 위한 대부분의 데이터는 단일 문장의 혐오 표현이며, 혐오 발언 검출 연구의 대부분이 단일 문장에 대한 혐오 표현을 분류하고있음.
- 혐오 단어를 포함하지 않더라도 특정 맥락이 제공될때, 혐오 발언으로 간주될 수 있는 경우가 존재함. 그러나 맥락-혐오 발언 쌍으로 구성된 적절한 데이터 셋이 거의 없음.

**문제 해결의 아이디어**
- 모델 훈련을 위한 데이터 수 부족을 해결하기 위해, 데이터 Augmentation 방법을 적용하였음. 맥락 문장과 혐오 발언에 대해 GPT-neo 모델을 fine-tuning 하고 유사한 문장을 생성하여 모델 훈련에 사용함.

**아이디어 적용 결과**
- 모델의 전반적인 분류 성능은 높게 나타남. 그러나 혐오 발언의 출처에 따라서 성능 저하를 보였음.
- 모델 훈련에 Augmented 맥락 문장이 포함된 경우, Original 문장과 동일한 출처의 문장 분류 성능은 유지되었지만, 전혀 다른 출처의 혐오 문장 분류 성능은 저하되었음.

**프로젝트 논의**
- 본 프로젝트에서 사용할 수 있었던 맥락 문장의 출처(도메인)가 지나치게 제한적이었으며, GPT를 활용하여 Augmented된 문장을 포함하여 훈련된 모델이 전혀 다른 출처의 혐오 문장 분류에는 큰 도움이 되지 못함.
- 혐오 표현의 검출에서 맥락 문장을 적절하게 반영하기 위해서는 무엇보다 다양한 데이터 도메인에서 맥락-혐오 표현으로 구성된 데이터의 확보가 우선시 되어야 할 것으로 생각됨.
- 그러나 일정 수준의 Original 데이터를 얻을 수 있다면, 상당한 비용과 시간을 들여서 수집하고 annotation한 데이터를 GPT를 활용하여 augmentation 함으로써 모델의 성능에 도움을 줄 수 있는 적절한 데이터를 더 확보할 수 있을 것임.

### *Notebook file explanation*

|Filename|Contents|
|---|---|
|/data|데이터는 일부만 업로드 되어있습니다.|
|/utils|Model train에 쓰인 Custom 유틸리티 함수|
|01_Preprocessing_data.ipynb|Data Collection & Preprocessing|
|02-1_GPT-neo(parents).ipynb|fine-tuning With 맥락(parent) 문장&생성|
|02-2_GPT-neo(toxic).ipynb|fine-tuning With 혐오(text) 문장&생성|
|03_Electra_model.ipynb|Train & Evaluation with Electra model|
|04_DistilBERT_model.ipynb|Train & Evaluation with DistilBERT model|
|05_BentoML.ipynb|Model Serve with BentoML tutorial|
|06_Inference_time.ipynb|Evaluate model inference time|

