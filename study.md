## Intro
- 고유값(Eigenvalue)과 고유벡터(Eigenvector)는 선형대수(Linear Algebra)에서 가장 중요한 이론 중 하나이며, 많은 머신러닝 이론에서 사용됨   
- 역사적으로 이차형식(Quadratic Forms) 및 미분방정식(Differential Equations) 이론으로 부터 발전      
- 전통적으로는 수학적 미분방정식을 풀기 위해 도입되었지만, 최근에는 인공지능을 포함한 머신러닝에서 사용되고 있어 중요도가 높아짐   
   
- 행렬은 **선형 변환**    
ㄴ 변환 후에도 원점의 위치가 변하지 않음   
ㄴ 변환 후에도 격자의 형태가 직선을 유지   
ㄴ 격자 간의 간격이 균등   
[위키백과 : 선형변환](https://ko.wikipedia.org/wiki/%EC%84%A0%ED%98%95_%EB%B3%80%ED%99%98)   


## 고유값과 고유벡터란?

정방행렬(Square Matrix)인 선형변환 A에 의한 변환 결과가 자기 자신의 상수배가 되는 0이 아닌 벡터를 "고유벡터라고 하며, 이 상수배의 값을 "고유값"이라고 함.   

수식으로 표현하면, n X n 정방행렬 A에 대해  $$Av =\lambda v $$ 를 만족하는 0이 아닌 열벡터 $v$를 고유벡터, 상수 $\lambda$를 고유값이라고 정의함.   

![enter image description here](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https://blog.kakaocdn.net/dn/JSW6W/btqEa03CDsu/yhMk1iWVdloNTcXG2qmWEK/img.png)

* 기하학적으로 고유벡터는 벡터에 선형변환을 적용할 경우 방향이 변하지 않는 벡터.   
* 크기만 변화하는데 이 때 변화되는 크기가 고유값.    

![enter image description here](https://blog.kakaocdn.net/dn/sn7gt/btqGuNy6UxN/mvtsrR1Tx6VLtf0vJtVr8K/img.gif)

ㄴ 빨간색 벡터 : 벡터의 방향과 크기가 모두 변함. 고유벡터가 아님.   
ㄴ 파란색 벡터와 분홍색 벡터: 방향은 변하지 않고 크기만 변함. 고유벡터임.   

고유값과 고유벡터를 찾아보자. $$ (A - \lambda I)v = 0 $$   
여기서 $I$는 단위행렬(Identity Matrix)이고 $A - \lambda I$ 의 역행렬이 존재한다면 $v$가 0이 되어버림. 0이 아닌 벡터를 찾으려고 하기 때문에  $A - \lambda I$ 의 역행렬이 존재하지 않아야함. 즉,  $$ det(A - \lambda I) = 0 $$    

위의 식을 특성방정식(Characteristic Equation)이라고 함. 특성방정식에서 $\lambda$ 의 값을 계산할 수 있고 이를 통해 고유벡터를 구함   

> 왜 우리는 고유벡터와 고유값을 찾아야할까?   

![enter image description here](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https://blog.kakaocdn.net/dn/s1lMz/btqEd8Zzqcv/f6oLnYDFd3SbONNy1fOc0k/img.png)
$Av$를 계산하려면 총 4번의 연산이 필요하지만, $\lambda v$는 2번의 연산으로도 충분함. Matrix가 커지면 차이가 더 커지게됨. 고유값과 고유벡터를 사용하는 것이 메모리 측면에서도 효율적이고 회전 연산 등을 할 때도 쉽고 빠르게 가능.   


## 고유값 분해(Eigen Decomposition)
고유값 분해는 고유값과 고유벡터로부터 유도되는 고유값 행렬과 고유벡터 행렬에 의해 분해될 수 있는 행렬의 표현.   
선형 독립을 만족하는 정방행렬 A는 고유값 분해에 의해 다음과 같이 분해됨.   
$$A = P\Lambda P^{-1} $$   
여기서 P는 정방행렬 A의 고유벡터들을 열벡터로 가지는 행렬이고 $\Lambda$는 고유값을 대각원소로 가지는 대각행렬.   

이를 정방행렬 A의 대각화(Diagonalization) 또는 고유값 분해(Eigen Decomposition)라고 함.   

[참고](https://angeloyeo.github.io/2020/11/19/eigen_decomposition.html)


## 특이값 분해(Sigular Value Decomposition)
특이값 분해는 임의의 고유값 분해를 직사각형행렬(Non-square Matrix)에 대해 일반화하는 방법으로 데이터의 차원을 축소하는 방법으로 사용됨.   
주성분분석(PCA, Principle Component Analysis)와 같은 분야에서 특이값 분해가 흔하게 쓰임.   

데이터 전체 차원보다 낮은 차원으로 근사시켜 적합(fit) 시킬 수 있는 공간을 찾는 것!   
d차원인 A 행렬을 특이값 분해를 통해 k차원(d>k)으로 축소시킨 B행렬을 찾아줌.   

![enter image description here](https://losskatsu.github.io/assets/images/svd/svd01.jpg)     

행렬 A의 크기가 $n \times d$ 라면 d차원에서 n개의 점이 있다고 생각할 수 있음.   
따라서 A에 대한 차원축소란 n개의 점을 표현할 수 있는 작은 차원인 d차원-부분공간을 찾는 문제라고 볼 수 있음.     

부분공간은 어떻게 구할까? 바로 데이터와 부분 공간으로부터의 수직거리를 최소화 시키는 것!   
직선거리의 최소화는 회귀분석에서 나온 제곱합(sum of squre)를 최소화 시키는 것.   
제곱합을 구해야 하므로 $AA^T, A^TA$ 를 사용하는 것이고, 이에 특이값 분해에서는 원본행렬 $A$보다는 $AA^T, A^TA$를 사용   

![enter image description here](https://losskatsu.github.io/assets/images/svd/svd02.jpg)   

 $A^{T}A$는 A의 열의 내적 = A의 열과 원점과의 거리   
 $AA^T$는 A의 행의 내적 = A의 행과 원점과의 거리   

 
![enter image description here](https://losskatsu.github.io/assets/images/svd/svd04.jpg)    

 
$$ M =U \Sigma V ^T $$ㅍ
   
여기서 $U$와 $V$는 유니터리 행렬(Unitary Matrix)이며, $\Sigma$는 대각행렬임.   

유니터리 행렬은   
$$U^T = U^{-1}$$   

 를 만족하며, $U = MM^T, V = M^TM$   
 
 

$M =U \Sigma V ^T$를  시각화하면   
![enter image description here](https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/Singular_value_decomposition_visualisation.svg/440px-Singular_value_decomposition_visualisation.svg.png)   




## ED가 머신러닝에서 어떻게 쓰일까?
우리가 다루는 데이터는 Symmetric positive definite matrix인 경우가 대부분.   
이에 feature(예. 키, 몸무게 등)를 row로, data item(예. 사람)을 col으로 갖는 행렬A가 있다면 $AA^{T}$ 는 feature by feature matrix가 될꺼고, $A^{T}A$는 data item by data item matrix가 될 것임.   


## 딥러닝에서 선형변환의 기하학적 의미

![enter image description here](https://blog.kakaocdn.net/dn/Styab/btqUuMgddAN/YKSGsN6EXX3iUaz6LeCouK/img.gif)

원래의 모눈종이가 standard **basis**들이라고 하면 ([1,0].T , [0,1].T)     
모눈종이가 점점 기울어진 평행사변형이 되는 것이 **linear transform** (선형변환) 이고,     
곡선으로 꾸겨지는 부분은 **non linear** 함수를 사용했을 때의 모습.   
(이 때 0 부분은 거의 그대로 유지되는 모습을 볼 수 있음.)     
흐르는 것은 **bias** 를 표현한 것.     
이러한 일련의 과정이 딥러닝의 node에서 이루어지고 있는 모습을 시각화 한 것.   


## 참고) 행렬에서 랭크란?

행렬 A의 열들 중에서 선형 독립인 열들의 최대 개수를 Rank라고 함. 행에 대해서 나타나는 공간의 차원과 같음.   
행(열) 전체가 0이 아닌 행(열)의 갯수를 행렬의 계수 = Rank 라고 함.   


$$A = \begin{bmatrix}
1 & 0 & 1 & 0\\
-2 & -3 & 1 & 0 \\
3 & 3& 0 & 0
\end {bmatrix}$$

얼핏 보면 네번째 열의 값이 모두 0이기 때문에 이를 제외한 나머지 1~3열은 각각 다른 값을 가지고 있는 것으로 보임.   
하지만 A행렬의 경우 랭크는 2. 첫번째 열과 두번째 열은 선형 독립이지만, 세번째 열은 첫번째 열에서 두번째 열을 빼주면 세번째열이 됨. 즉 선형 종속(1차종속)인 관계가 되어 Rank는 2   

반대로 어떤 행(열)이 다른 행(열)들의 선형 조합으로 표현될 수 없다면 선형 독립이라고 함.   

머신러닝에서 랭크가 의미하는 바는 다음과 같은 예시로 나타낼 수 있음.   

키, 몸무게 등의 feature(column)으로 이루어진 데이터셋이 있을 때, 극단적으로 V2, V3, V4가 모두 V1에 선형 의존적인 관계일 수 있음. 이런 상황에서는 피쳐의 갯수는 많지만 rank가 줄어들게 됨.   

rank는 얻을 수 있는 정보양과 관련이 있음. rank가 작으면 데이터셋에서 얻을 수 있는 정보가 줄어들기 때문에 ML모델에 상당한 방해를 하고 영향을 끼치게 됨.   



> Reference
[고유벡터와 고유값](https://junklee.tistory.com/87)   
[특성방정식](https://junklee.tistory.com/89?category=937502)   
[특이값 분해](https://losskatsu.github.io/linear-algebra/svd/#)   
[벡터 직교성의 공학적 의미](https://satlab.tistory.com/37)   
[딥러닝에서 선형변환](https://jxnjxn.tistory.com/47)   
[행렬에서 Rank란?](https://blog.naver.com/sw4r/221416614473)   
[공돌이의 수학노트 정리](https://angeloyeo.github.io/2020/09/07/basic_vector_operation.html)   
