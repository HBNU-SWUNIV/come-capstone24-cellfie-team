# 한밭대학교 컴퓨터공학과 Cellfie팀

**팀 구성**
- 20191744 이성화 
- 20191779 오종권
- 20217138 이성원

## **Teamate** Project Background

### 필요성
- 용존 산소 부족은 독극물을 제외하면 어류 폐사의 가장 중요한 환경 요인이다.
- 수온과 어류의 종류에 따라 용존 산소 농도가 변동한다.
- 용존 산소 농도는 어류의 산소 소비량과 밀접하게 관련된다.
- 어류의 산소 섭취량은 다음 요소에 따라 달라진다:
  - 어류의 크기
  - 사료 섭취량
  - 스트레스 수준
- 산소 섭취량은 어류의 건강과 성장에 직접적인 영향을 미친다.

### 기존 해결책의 문제점
- 상시 모니터링 및 산소 발생기를 항상 사용하므로 전력 소모가 크다.
- 현재 고가의 용존 산소 측정 센서가 설치된 양식장은 드물다.
- 주로 정부 지원을 받은 양식장에만 센서가 설치되어 있다.
- 이로 인해 용존 산소 농도 감시와 이상 상황 발생 시 적절한 대응이 부족할 수 있다.

  
## System Design

### System Requirements

<p align="center">
  <img src="https://github.com/user-attachments/assets/a537b0fa-a13d-4da4-8447-70e6c9bc7f70" alt="image" width="900"/>
</p>


1. **백엔드 서버**는 Spring Boot로 구성되어 있으며, 1분마다 외부 API를 통해 기상 정보와 양식장 센서 데이터를 가져옵니다. 양식장에서는 수온, 용존산소, 염도, pH 정보를 수집하고, 바다누리 API를 통해 바다의 수온, 풍향, 풍속, 염도, 기온, 기압, 조위 정보를 수집한다.
   
2. **인공지능 모델**로 수집한 데이터를 전송하여 30분 후의 용존산소량을 예측하고, 예측 결과를 서버로 전송합니다. Triton을 이용하여 ML 서버를 구축하고 메인 백엔드 서버와 통신한다.
   
3. **데이터베이스**는 MariaDB로 구축되며, 외부 데이터와 용존산소 예측값을 1분마다 저장한다.
   
4. **프론트엔드**는 예측된 용존산소량, 현재 용존산소량, 수온 등의 데이터를 받아, 그래프 등으로 유저에게 시각화하여 보여준다.

<p align="center">
  <img src="https://github.com/user-attachments/assets/bfaf4bd1-d2e1-4f18-8490-e9cd34c04f24", alt="image", width="1500"/>
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/44e085eb-20b0-4918-a736-796541cfef37", alt="image", width="1500"/>
</p>

- **액터:** 사용자
- **목표:** 양식장의 데이터를 차트와 텍스트 형태로 확인하고, 데이터에 대해 더 깊이 있는 정보를 얻기 위해 상호작용한다.
- **사전 조건:** 사용자가 시스템에 로그인하고 데이터 액세스 권한을 가지고 있어야 한다.

### 기본 흐름:
1. **대시보드 접근:**
   - 사용자는 양식장 데이터 대시보드에 접근한다.
   - 시스템은 실시간 양식장 데이터(수온, 염도, pH농도, 용존 산소, 용존 산소 예측량)를 불러온다.
   
2. **차트 표시:**
   - 시스템은 가져온 데이터를 차트 형태로 사용자에게 표시한다.
   
3. **차트 상호작용:**
   - 사용자는 차트의 특정 데이터 포인트를 선택하여 상세 데이터를 요청한다.
   - 시스템은 선택된 데이터에 대한 세부 정보를 제공한다.
   
4. **세부 정보 확인:**
   - 시스템은 사용자가 선택한 데이터 포인트에 대한 세부 정보를 텍스트 형식으로 표시한다.

5. **데이터 업데이트 및 반복 조회:**
   - 사용자가 추가 데이터를 요청하면, 시스템은 실시간으로 데이터를 업데이트하고 사용자에게 표시한다.

### 대안 흐름:
- **데이터 로딩 실패:** 데이터를 로드하는 과정에서 오류가 발생할 경우, 시스템은 오류 메시지를 사용자에게 표시하고, 다시 시도할 수 있는 옵션을 제공한다.

### 후조건:
- 사용자가 필요한 모든 데이터 정보를 성공적으로 확인한다.
- 사용자가 시스템을 종료하거나 다른 페이지로 이동한다.

<!--- <div align="center">![image](https://github.com/user-attachments/assets/7e227a46-2d69-43e1-9452-886a530401f2)</div>
<div align="center">![image](https://github.com/user-attachments/assets/768f7b7d-aaa7-40f3-b171-153f1bf23948)</div>
    
- 서버에서 받을 두 종류의 데이터이다. 이 두 데이터를 시계열 데이터 형태로 전처리를 진행한 후 병합해주고, 병합한 데이터는 학습시킨 모델을 통해 특정 시간 이후의 예측한 용존산소량을 서버로 반환해준다. -->
    
## Case Study
  - ### Description
  
  
## Conclusion
  - ### OOO
  - ### OOO
  
## Project Outcome
- ### 2024년 대한전자공학회 하계 학술대회 
