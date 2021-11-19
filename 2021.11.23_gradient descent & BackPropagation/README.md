
## 손실함수(Loss function)
![](https://wikidocs.net/images/page/36033/%EC%86%90%EC%8B%A4%ED%95%A8%EC%88%98.PNG)


- 손실함수는 실제값과 예측값의 차이를 수치화해주는 함수
- 오차가 클수록 손실함수 값이 크고, 오차가 작으면 손실함수의 값은 작음
- 회귀에서는 MSE, 분류에서는 Cross-Entropy를 손실함수로 사용
 -- MSE :  
![](https://wikidocs.net/images/page/24987/mse.PNG)
-- 크로스 엔트로피 : 
![](https://wikidocs.net/images/page/24987/%ED%81%AC%EB%A1%9C%EC%8A%A4%EC%97%94%ED%8A%B8%EB%A1%9C%ED%94%BC.PNG)
- 손실함수의 값을 최소화 하는 가중치 w와 편향 b를 찾는 것이 딥러닝의 학습 과정


## 옵티마이저(Optimizer)
![](https://wikidocs.net/images/page/36033/%EC%97%AD%EC%A0%84%ED%8C%8C_%EA%B3%BC%EC%A0%95.PNG)

- 손실함수 값을 줄여나가며 학습하는 방법은 어떤 옵티마이저를 사용하는지에 따라 달라짐

1) **배치 경사 하강법(Batch Gradient Descent)**
2)  **확률적 경사 하강법(Stochastic Gradient Descent, SGD)**
3)  **미니 배치 경사 하강법(Mini-Batch Gradient Descent)**
4) **모멘텀(Momentum)**
5)  **아다그라드(Adagrad)**
6)  **알엠에스프롭(RMSprop)**
7)  **아담(Adam)**
* 참고 사이트  : https://wikidocs.net/36033

## 경사하강법(Gradient Descent)
- 경사하강법은 신경망 안의 가중치 조합을 모두 계산하려면 비용이 많이 들기 때문에 효율적으로 계산을 하기 위해 고안된 방법

### 선형회귀에서 이해하는 경사하강법

![](https://t1.daumcdn.net/cfile/tistory/990E6F3B5C76075D37)

: 기온 x 어묵판매량에 대한 데이터로, 기온이 상승할수록 어묵판매량이 감소함
ㄴ 선형회귀로 모델링 가능해보임

![](https://t1.daumcdn.net/cfile/tistory/996489395C7608B834)

선형회귀의 목적인 위와 같은 직선을 찾는 법을 살펴보자.

1. w와 b를 임의로 설정한 일차함수와 데이터 사이의 평균제곱오차(MSE)를 구한다.
![](https://t1.daumcdn.net/cfile/tistory/997372335A0520F312)
ㄴ MSE를 비용함수(cost function)이라고 부름

![](https://t1.daumcdn.net/cfile/tistory/99ED93385C764D6222)


2. MSE(비용함수)를 최소가 되게 하는 w와 b를 찾는 것이 목표
ㄴ w와 b에 대한 비용함수는 아래와 같은 2차함수 형태가 됨

![](https://t1.daumcdn.net/cfile/tistory/990C103A5C764DDE25)

![](https://t1.daumcdn.net/cfile/tistory/997774505C7738DC02)

3. 비용함수를 최소로 하는 w와 b를 찾기 위해 임의로 w값을 하나 선정함(초기값 부여)
ㄴ 운이 좋으면 초기값을 최적값으로 바로 찾겠지만, 그렇지 않을 확률이 큼
![](https://t1.daumcdn.net/cfile/tistory/99FF5C3B5C773F760B)

4. 최적의 w를 찾기 위해서는 비용함수를 w에 대해 편미분해주고, 학습률(learning rate)라고 불리는 파라미터 a를 곱한것을 초기 설정된 w에서 빼준다.
![](https://t1.daumcdn.net/cfile/tistory/996243395C774C461D)
ㄴ 이 때, 학습률 파라미터(a)는 적절한 값으로 사용자가 설정함.  (a는 양수인 실수)

5. 4의 과정을 반복
![](https://t1.daumcdn.net/cfile/tistory/998994505C7750611E)
ㄴ 비용함수를 w에 대해 편미분하면 현재 w 위치에서의 기울기와 같음.
ㄴ 예를들어 초기 w값(w0)을 1이라고 하고 a를 2라고 하고 w0일때 기울기를 -2라고 가정하면 다음 w1 = 1 - (2*(-2)) = 5가 됨. 즉, 두번째 그래프처럼 w가 오른쪽으로 이동하게 됨. 
ㄴ 이와 같은 과정을 반복해주면 최적의 w값을 찾을 수 있게 됨(기울기가 0이 될때까지)
ㄴ 경사가 점점 감소하는 현상을 이용하기 때문에 경사감소법이라고 부름

6. 최적의 b도 같은 방법으로 찾아줌
7. 최적의 w와 b를 찾아가는 과정은 아래와 같음
![](https://t1.daumcdn.net/cfile/tistory/99A9FB4C5C7755362D)
ㄴ초기값으로 설정된 w와 b는 데이터를 제대로 반영하지 못하지만, 경사감소법을 사용해 최적의 일차함수를 찾아냄

### 경사하강법의 특징
-  경사하강법을 사용하면 신경망의 계산속도를 빠르게 하며, 복잡한 수식에서 잘 작동함
- 데이터가 불완전해도 유도리 있게 작동함
-  그러나, 오차를 줄이며 조금씩 내려가는 과정에서 너무 많이 이동하면 최저점을 지나게 되고,  너무 조금씩 이동하면 이동회수가 많아져 최저점을 찾지 못할수도 있음 
- 또한, 차원이 여러개라고 한다면 엉뚱한 최저점을 찾을수도 있음


## 역전파(BackPropagation)

- 역전파 알고리즘은 input과 output을 알고있는 상태에서 신경망을 학습하는 방법
-  농구 자유투에 빗대어 설명하자면, 자유투를 던지는 과정은 순전파(Feed Forward)라고 할 수 있고, 던진 공이 어느 지점에 도착했는지 확인하고 던질 위치를 수정하는 과정을 역전파(Backpropagation)이라고 할 수 있음
- 결과에 영향을 많이 미친 노드(뉴런)에 더 많은 오차를 돌려주게 되며, 역전파 알고리즘은 앞서 배운 **경사하강법**을 사용해 가중치를 업데이트

![Image result for backpropagation](https://i.stack.imgur.com/H1KsG.png)
ㄴ input이 들어오는 방향인 순전파로 output layer에서 결과값이 나오게 됨. 결과값은 오차를 가지게 되는데, 역전파는 이 오차를 다시 역방향인 hidden layer와 input layer로 내보내며 가중치를 계산해 output에서 발생했던 오차를 적용시킴

한 번 돌리는 것을 1 epoch라고 하며 epoch를 늘릴수록 가중치가 계속 업데이트(학습)되어 점점 오차가 줄어들게 됨

### 예제) 순전파
![](https://wikidocs.net/images/page/37406/backpropagation_2.PNG)
![](https://latex.codecogs.com/gif.latex?%5Clarge%20o_%7B1%7D%3Dsigmoid%28z_%7B3%7D%29%3D0.60944600)
![](https://latex.codecogs.com/gif.latex?%5Clarge%20o_%7B2%7D%3Dsigmoid%28z_%7B4%7D%29%3D0.66384491)

실제값 0.4와 예측값 0.60944600, 실제값 0.6과 예측값 0.66384491의 오차를 계산하기 위해 손실함수(loss function)를 MSE로 사용. 식에서 실제값은 target, 순전파를 통해 나온 예측값은 output으로 표현하고, 각 오차를 모두 더하면 전체오차 E_total이 됨

![](https://latex.codecogs.com/gif.latex?%5Clarge%20%5Cbegin%7Balign*%7D%20E_%7Bo1%7D%3D%5Cfrac%7B1%7D%7B2%7D%28target_%7Bo1%7D-output_%7Bo1%7D%29%5E%7B2%7D%3D0.02193381%20%5C%5C%20E_%7Bo2%7D%3D%5Cfrac%7B1%7D%7B2%7D%28target_%7Bo2%7D-output_%7Bo2%7D%29%5E%7B2%7D%3D0.00203809%20%5C%5C%20E_%7Btotal%7D%3DE_%7Bo1%7D&plus;E_%7Bo2%7D%3D0.02397190%20%5Cend%7Balign*%7D)


### 역전파 1단계
순전파가 입력층에서 출력층으로 향한다면 역전파는 반대로 출력층에서 입력층 방향으로 계산하며 가중치를 업데이트함. 출력층 바로 이전의 은닉층을 N층이라고 하면, 출력층과 N층 사이의 가중치 업데이트를 역전파 1단계, N층과 N층의 이전층 사이의 가중치 업데이트를 역전파 2단계라고 함

![](https://wikidocs.net/images/page/37406/backpropagation_3.PNG)

1단계에서 업데이트 해야 할 가중치는 w5~w8 총 네개. w5에 대해 업데이트를 하기 위해서는 경사하강법을 수행. 전체 오차에 대해 w5로 편미분을 해야하며, 미분의 연쇄법칙으로 계산

![](https://latex.codecogs.com/gif.latex?%5Clarge%20%5Cbegin%7Balign*%7D%20%5Cfrac%7B%5Cpartial%20E_%7Btotal%7D%7D%7B%5Cpartial%20w_%7B5%7D%7D%3D%20%5Cfrac%7B%5Cpartial%20E_%7Btotal%7D%7D%7B%5Cpartial%20o_%7B1%7D%7D%20%5Ctimes%20%5Cfrac%7B%5Cpartial%20o_%7B1%7D%7D%7B%5Cpartial%20z_%7B3%7D%7D%20%5Ctimes%20%5Cfrac%7B%5Cpartial%20z_%7B3%7D%7D%7B%5Cpartial%20w_%7B5%7D%7D%20%5Cend%7Balign*%7D)
순서대로 계산하면 0.02592286이 나오며, 이를 앞서 배운 경사 하강법을 통해 가중치를 업데이트. 파하이퍼파라미터에 해당하는 학습률(learning rate) a는 0.5라고 가정

![](https://latex.codecogs.com/gif.latex?%5Clarge%20%5Cbegin%7Balign*%7D%20w_%7B5%7D%5E%7B&plus;%7D%3Dw_%7B5%7D-%5Calpha%20%5Cfrac%7B%5Cpartial%20E_%7Btotal%7D%7D%7B%5Cpartial%20w_%7B5%7D%7D%3D0.45-%200.5%20%5Ctimes%200.02592286%3D0.43703857%20%5Cend%7Balign*%7D)
같은 방식으로 나머지 가중치에 대해서도 업데이트 시행


### 역전파 2단계
![](https://wikidocs.net/images/page/37406/backpropagation_4.PNG)

이번 단계에서 계산할 가중치는 w1~w4이며, 원리 자체는 역전파 1단계에서와 동일함.
![](https://latex.codecogs.com/gif.latex?%5Clarge%20%5Cbegin%7Balign*%7D%20w_%7B1%7D%5E%7B&plus;%7D%3Dw_%7B1%7D-%5Calpha%20%5Cfrac%7B%5Cpartial%20E_%7Btotal%7D%7D%7B%5Cpartial%20w_%7B1%7D%7D%3D0.1-%200.5%20%5Ctimes%200.00080888%3D0.29959556%20%5Cend%7Balign*%7D)

같은 방식으로 나머지 가중치도 계산해줌

![](https://wikidocs.net/images/page/37406/backpropagation_5.PNG)

업데이트 된 가중치를 사용해 다시 순전파를 진행하고 오차가 감소했는지 확인
기존의 전체 오차는 0.02397190이였는데, 한번의 역전파 이후 오차를 계산하니 0.02323634가 나옴. 오차가 감소했으며, 인공 신경망의 학습은 오차를 최소화하는 가중치를 찾는 목적으로 순전파와 역전파를 반복하는 것


### 역전파 알고리즘의 한계
-  경사하강법의 한계와 동일하게 항상 전역 최소값인 global minimum을 찾는다는 보장이 없음. 


### Reference
- 경사하강법 : https://bskyvision.com/411
- 역전파 : https://wikidocs.net/37406
- 경사하강법(영상) : https://www.youtube.com/watch?v=IHZwWFHWa-w
- 역전파(영상) : https://www.youtube.com/watch?v=Ilg3gGewQ5U

