# 한밭대학교 컴퓨터공학과 Cellfie팀

**팀 구성**
- 20191744 이성화 
- 20191779 오종권
- 20217138 이성원

## <u>Teamate</u> Project Background
- ### 필요성
  - 용존 산소 부족은 독극물을 제외하면 어류 폐사의 가장 중요한 환경 요인임.
  <br/>- 수온과 어류의 종류에 따라 용존 산소 농도가 변동함.
  <br/>- 용존 산소 농도는 어류의 산소 소비량과 밀접하게 관련됨.
  <br/>- 어류의 산소 섭취량은 다음에 따라 달라짐:
      <br/>     어류의 크기, 사료 섭취량, 스트레스 수준
  <br/>- 산소 섭취량은 어류의 건강과 성장에 직접적인 영향을 미침.
### 기존 해결책의 문제점
  - 상시 모니터링 및 산소발생기를 항상 사용하므로 전력이 많이 소모됨.
  <br/>- 현재 고가의 용존 산소 측정 센서가 설치된 양식장은 드물음.
  <br/>- 주로 정부 지원을 받은 양식장에만 센서가 설치됨.
  <br/>- 이로 인해 용존 산소 농도 감시와 이상 상황 발생 시 적절한 대응이 부족할 수 있음. 
  
## System Design
  - ### System Requirements
    ![image](https://github.com/user-attachments/assets/a537b0fa-a13d-4da4-8447-70e6c9bc7f70)
     <br/>① : 백엔드 서버는 Springboot로 구성하여 1분마다 외부 API를 통해 기상정보와 양식장 센서의 데이터를 가져온다. 양식장에서는 수온, 용존산소, 염도, pH의 정보를 수집하며, 바다누리API를 통해 바다의 수온, 풍향, 풍속, 염도, 기온, 기압, 조위에 대한 정보를 수집한다. 
     <br/>② : 수집한 데이터를 인공지능 모델로 전송한다. 이를 토대로 30분 후의 용존산소량을 예측하여 서버로 전송한다. 이때 Triton이용하여 ML서버를 구축하고 메인 백엔드 서버와 통신한다.
     <br/>③ : 1에서 수집안 외부 데이터와 용존산소량의 예측값을 1분마다 데이터베이스에 저장한다. 이때 데이터베이스는 MariaDB로 구축한다.
     <br/>④ : 용존산소량의 예측값 및 현재 용존산소량 및 수온 등의 데이터를 프론트엔드로 전송한다. 프론트엔드에서는 그래프 등의 형식을 통해 유저에게 데이터를 보여준다. 

    ![image](https://github.com/user-attachments/assets/bfaf4bd1-d2e1-4f18-8490-e9cd34c04f24)
    ![image](https://github.com/user-attachments/assets/44e085eb-20b0-4918-a736-796541cfef37)
     <br/>-  액터: 사용자
     <br/>-  목표: 양식장의 데이터를 차트와 텍스트 형태로 확인하고, 데이터에 대해 더 깊이 있는 정보를 얻기 위해 상호작용한다.
     <br/>-  사전 조건: 사용자가 시스템에 로그인하고 데이터 액세스 권한을 가지고 있어야 한다.
     <br/>-  기본 흐름:
      <br/>1. 대시보드 접근:
        <br/>● 사용자는 양식장 데이터 대시보드에 접근한다.
        <br/>● 시스템은 실시간 양식장 데이터(수온, 염도, pH농도, 용존 산소, 용존 산소 예측량)를 불러온다.
      <br/>2. 차트 표시:
        <br/>● 시스템은 가져온 데이터를 차트 형태로 사용자에게 표시한다.
      <br/>3. 차트 상호작용:
        <br/>● 사용자는 차트의 특정 데이터 포인트를 선택하여 상세 데이터를 요청한다.
        <br/>● 시스템은 선택된 데이터에 대한 세부 정보를 제공한다.
      <br/>4. 세부 정보 확인:
        <br/>● 시스템은 사용자가 선택한 데이터 포인트에 대한 세부 정보를 텍스트 형식으로 표시한다.
      <br/>1. 데이터 업데이트 및 반복 조회:
        <br/>● 사용자가 추가 데이터를 요청하면, 시스템은 실시간으로 데이터를 업데이트하고 사용자에게 표시한다.
         <br/>-  대안 흐름:
        <br/>● 데이터 로딩 실패: 데이터를 로드하는 과정에서 오류가 발생할 경우, 시스템은 오류 메시지를 사용자에게 표시하고, 다시 시도할 수 있는 옵션을 제공한다.
         <br/>-   후조건:
        <br/>● 사용자가 필요한 모든 데이터 정보를 성공적으로 확인한다.
        <br/>● 사용자가 시스템을 종료하거나 다른 페이지로 이동한다.

    ![image](https://github.com/user-attachments/assets/7e227a46-2d69-43e1-9452-886a530401f2)
    ![image](https://github.com/user-attachments/assets/768f7b7d-aaa7-40f3-b171-153f1bf23948)
        <br/>- 서버에서 받을 두 종류의 데이터이다. 이 두 데이터를 시계열 데이터 형태로 전처리를 진행한 후 병합해준다. 병합한 데이터는 학습시킨 모델을 통해 특정 시간 이후의 예측한 용존산소량을 서버로 반환해준다.
    
## Case Study
  - ### Description
  
  
## Conclusion
  - ### OOO
  - ### OOO
  
## Project Outcome
- ### 2024년 대한전자공학회 하계 학술대회 
