## 개요
`인공지능을 활용한 자동차 외관 부품 불량 탐지 플랫폼`에 사용된 객체 탐지 모델 입니다.

![샌드위치ai_페이지2](https://github.com/user-attachments/assets/5ffbc39b-dff8-4f3e-b79c-98ed8baeb6a7)


## 파일 소개
1. **IMG** <br>
   코드 테스트를 위한 예시 이미지
2. **best.pt** <br>
   프로젝트에 가장 최적화된 객체 탐지 모델
3. **test.py** & **restapi.py**
   배포에 사용된 코드

## 모델 소개

총 네 개의 클래스가 존재합니다. 단차, 장착 불량, 스크래치, 외관 손상입니다. 각 클래스 별 정확도와 훈련을 거치며 어떻게 발전했는지를 정리한 표입니다.

| 모델 정확도 |
| ----------- |
| ![image](https://github.com/user-attachments/assets/722ccf59-62f7-4c7f-81f2-5df6c6c3069e) |

| 정확도 변천사 |
| ------------- |
| ![image](https://github.com/user-attachments/assets/1df25815-f96d-4e57-bef4-d35943ce9bb0) |
