import styled from "styled-components";

export const ToggleContainer = styled.div`
  position: absolute;
  width: 50px;
  height: 24px;
  left: 96%;
  top: 2%;
  cursor: pointer;
  > .toggle-container {
    width: 50px;
    height: 24px;
    border-radius: 30px;
    background-color: rgb(233, 233, 234);
  }
  //.toggle--checked 클래스가 활성화 되었을 경우의 CSS를 구현
  > .toggle--checked {
    background-color: rgb(0, 200, 102);
    transition: 0.5s;
  }

  > .toggle-circle {
    position: absolute;
    top: 1px;
    left: 1px;
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background-color: rgb(255, 254, 255);
    transition: 0.5s;
  }
  > .toggle--checked {
    left: 27px;
    transition: 0.5s;
  }
`;

export const Desc = styled.div`
  text-align: right;
  margin: 10px;
  top: 5%;
  left: 92%;
  position: absolute;
`;
