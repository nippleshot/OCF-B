## OCF-B experiment implementation

#### Prerequisites for Importing

- Keras
- NumPy
- CuPy
- Pandas
- Scikit-learn
- Matplotlib
- Tqdm
- Colab (Preferred)



#### Launch Test

- Validation test (w/ MovieLens100K) : 

  *@main.py*

  ```python
  if __name__ == "__main__":
      negaCaseNum_list = [20000, 50000, 80000]
      experiment1_by_negaCaseNum(negaCaseNum_list, 10, "MovieLens100K_oneClass.csv", 943, 1682)
  ```

  

- Performance comparison test (w/ MovieLens100K) : 

  *@ZeroInjection.py*

  ```python
  if __name__ == "__main__":
    negaCaseNum_list = [20000, 50000, 80000]
    experiment2_by_negaCaseNum(negNumList)
  ```

  

#### Explanation of function

##### Notations

| Python Data Type                                             |
| ------------------------------------------------------------ |
| <span style="color:blue">***Integer***</span>                |
| <span style="color:red">***String***</span>                  |
| <span style="color:purple">***List***</span>                 |
| <span style="color: fuchsia">***Dictionary*** </span>        |
| <span style="color: green">***tf.keras.Model*** </span>      |
| <span style="color: goldenrod">***pandas.DataFrame***</span> |



##### main.py

- **`experiment1_by_negaCaseNum(...)`**

  - OCF-B 알고리즘 유효성 실험 실행 및 결과 출력

  - Parameters

    | Name                                                 | About                                            |
    | ---------------------------------------------------- | ------------------------------------------------ |
    | <span style="color:purple">`negaCaseNum_list`</span> | 실험해보고 싶은 부정사례 데이터 수량들           |
    | <span style="color:blue">`iterNum`</span>            | 원하는 OCF-B 알고리즘 진행 iteration 횟수        |
    | <span style="color:red">`fileName`</span>            | one class만 가지고 있는 데이터셋의 파일명+확장자 |
    | <span style="color:blue">`userLimit` </span>         | 데이터셋 내 최대 사용자 ID                       |
    | <span style="color:blue">`itemLimit` </span>         | 데이터셋 내 최대 아이템 ID                       |

  

- **`printGraph(...)`**

  - 훈련된 모델 성능의 변화를 그래프( X : <span style="color:blue">`iter`</span> , Y : <span style="color:red">`eval_index`</span> )로 형성 및 저장

  - Parameters

    | Name                                            | About                                                        |
    | ----------------------------------------------- | ------------------------------------------------------------ |
    | <span style="color:red">`eval_index`</span>     | evaluation metric 명                                         |
    | <span style="color:blue">`iter`</span>          | OCF-B 알고리즘의 iteration 횟수                              |
    | <span style="color:red">`dataName`</span>       | 결과 그래프 명                                               |
    | <span style="color:fuchsia">`eval_datas`</span> | 그래프로 출력할 결과 데이터<br/>*Example :* `{'n': 50000, 0: 0.905..., 1: 0.944.., 2: 0.958.. } ` |

    

##### OCF_B_V1.py

- **`genNegative(...)`**

  - <span style="color:purple">`added_np_train`</span> 데이터셋에 <span style="color:blue">`num`</span> 개의 부정사례 데이터를 추가시켜서 <span style="color:purple">`added_np_train`</span> 를 return함

  - Parameters

    | Name                                               | About                                                        |
    | -------------------------------------------------- | ------------------------------------------------------------ |
    | <span style="color:blue">`num`</span>              | 생성하고 싶은 부정사례 데이터 수량                           |
    | <span style="color:purple">`added_pair`</span>     | 부정사례 데이터 생성时，생성하면 안되는  [userID, movieID] 형태의 데이터 list |
    | <span style="color:purple">`added_np_train`</span> | 다음 iteration 훈련에 사용될 [userID, movieID, label, xx] 형태의 데이터 list |
    | <span style="color:blue">`userID_limit`</span>     | 데이터셋 내 최대 사용자 ID                                   |
    | <span style="color:blue">`itemID_limit`</span>     | 데이터셋 내 최대 아이템 ID                                   |



- **`evaluate(...)`**

  -  <span style="color:purple">`test`</span> 데이터를 활용하여 훈련된 모델 <span style="color:green">`model`</span>의 성능을 8개의 evaluation metric으로 측정하여 return함

  - Parameters

    | Name                                     | About                            |
    | ---------------------------------------- | -------------------------------- |
    | <span style="color:green">`model`</span> | 훈련된 Matrix Factorization 모델 |
    | <span style="color:purple">`test`</span> | 모델 성능 측정에 사용될 데이터셋 |



- **`run(...)`**

  - OCF-B 알고리즘 구현 및 실험 결과 데이터를 return 함 

  - 실험 결과 데이터 구축 :

    - OCF-B 알고리즘 실행 전 부터 후 까지 각 iteration 마다 **`OCF_B_V1.evaluate(...)`** 를 호출하여 얻은 8개의 evaluation metric를 각 metric의 대한 dictonary들에 나누어서 저장

  - Parameters

    | Name                                           | About                                                    |
    | ---------------------------------------------- | -------------------------------------------------------- |
    | <span style="color:blue">`numNegaCase`</span>  | OCF-B 알고리즘을 통해 생성하고 싶은 부정사례 데이터 수량 |
    | <span style="color:blue">`iter`</span>         | OCF-B 알고리즘 iteration 횟수                            |
    | <span style="color:red">`fileName`</span>      | one class만 가지고 있는 데이터셋의 파일명+확장자         |
    | <span style="color:blue">`userID_limit`</span> | 데이터셋 내 최대 사용자 ID                               |
    | <span style="color:blue">`itemID_limit`</span> | 데이터셋 내 최대 아이템 ID                               |



##### TrainingModule.py

- **`zeroInjectionMF(...)`**

  - `MovieLens100K_MF.csv` 데이터를 가지고 훈련시킨 Matrix Factorization 모델 저장 (성능 비교 실험 때 사용)

  - Parameters

    | Name                                            | About                                        |
    | ----------------------------------------------- | -------------------------------------------- |
    | <span style="color: goldenrod">`dataset`</span> | `MovieLens100K_MF.csv` 데이터                |
    | <span style="color:blue">`epo`</span>           | Matrix Factorization 모델 훈련 Epoch 수      |
    | <span style="color:blue">`NlatentFactor`</span> | Matrix Factorization 모델 latent factor 설정 |



- **`trainMFModel(...)`**

  - Matrix Factorization 모델 훈련 후 생성된 모델을 return 함 (유효성 실험 때 사용)

    - 이후 **`OCF_B_V1.run(...)`** 에서 train 데이터셋 내의 유효하지 않은 부정사례 제거 과정과 모델 성능 측정과정을 진행하기 위해 test, train 데이터셋도 같이return 함 

  - Parameters

    | Name                                            | About                                                        |
    | ----------------------------------------------- | ------------------------------------------------------------ |
    | <span style="color: goldenrod">`dataset`</span> | 훈련 데이터셋                                                |
    | <span style="color:blue">`epo`</span>           | Matrix Factorization 모델 훈련 Epoch 수                      |
    | <span style="color:blue">`NlatentFactor`</span> | Matrix Factorization 모델 latent factor 설정                 |
    | <span style="color:blue">`testSize`</span>      | <span style="color: goldenrod">`dataset`</span>의 몇 퍼센트를 테스트 데이터셋으로 활용할 건지 |

    

##### ZeroInjection.py

- **`makeInitialModel(...)`**

  - Zero Injection 방법을 진행하기 위해 <span style="color:red">`fileName`</span> 로 훈련시킨 초기 Matrix Factorization 모델 구축

  - Parameter

    | Name                                      | About                  |
    | ----------------------------------------- | ---------------------- |
    | <span style="color:red">`fileName`</span> | `MovieLens100K_MF.csv` |



- **`save_ZeroPredicted()`**
  - 초기 Matrix Factorization 모델를 활용하여 label이 0인 데이터에 대해 예측. 이후 예측값 크기 순으로 데이터 정렬 및  `MovieLens100K_ZeroSorted.csv` 로 저장



- **`get_NegativeCase(...)`**

  - 최종적으로 Zero로 Injection할 <span style="color:blue">`num`</span> 개 부정사례 데이터 불러오기

    - <a href="http://pike.psu.edu/publications/icde16.pdf">(Hwang et al., 2016)</a>에서의 θ parameter 설정 부분

  - Parameter

    | Name                                  | About                                              |
    | ------------------------------------- | -------------------------------------------------- |
    | <span style="color:blue">`num`</span> | 최종적으로 Zero로 Injection할 부정사례 데이터 수량 |



- **`experiment2_by_negaCaseNum(...)`**

  - Zero Injection 방법을 통해 생성한 부정사례 x개와 실제 긍정사례로 구성된 데이터셋으로 Matrix Factorization 모델 훈련 및 모델 성능 출력

  - Parameter

    | Name                                                 | About                                  |
    | ---------------------------------------------------- | -------------------------------------- |
    | <span style="color:purple">`negaCaseNum_list`</span> | 실험해보고 싶은 부정사례 데이터 수량들 |

    

##### Data Package

###### MovieLens100K

- `MovieLens100K_MF.csv`  : 
  - ***format*** : user_id | item_id | label | time (ignore)
    - user_id : [1,943]
    - item_id : [1,1682]
    - label  = `1` : MovieLens100K 원본 데이터셋에서 평점 4~5 데이터
    - label  = `0` : MovieLens100K 원본 데이터셋에서 평점 1~3 데이터 + Missing value 데이터
  - ***내용*** : MovieLens100K 원본데이터셋에서 실험에 필요한 형식으로 변환시킨 데이터
- `MovieLens100K_oneClass.csv`  : 
  - ***format*** : user_id | item_id | label | time (ignore)
  - ***내용*** : `MovieLens100K_MF.csv` 중 label = `1` 인 데이터 밖에 없음
- `MovieLens100K_ZeroList.csv`  : 
  - ***format*** : user_id | item_id | label | time (ignore)
  - ***내용*** : label  =  `MovieLens100K_MF.csv` 중 label = `0` 인 데이터 밖에 없음
- `MovieLens100K_ZeroResult.csv`  : 
  - ***format*** : user_id | item_id | predicted_value
  - ***내용*** : **`ZeroInjection.save_ZeroPredicted()`** 실행时,  `MovieLens100K_ZeroList.csv` 의 user_id | item_id 에 대한 예측값들 중간 저장
- `MovieLens100K_ZeroSorted.csv`  : 
  - ***format*** : user_id | item_id | predicted_value
  - ***내용*** :  `MovieLens100K_ZeroResult.csv` 의 예측값 크기 순으로 정렬한 데이터



###### Epinions

- `Epinions_oneClass.csv` 
  - ***format*** : user_id | item_id | label | time (ignore)
    - user_id : [1,49290]
    - item_id : [1,139738]
    - label  = `1` : Epinions 원본 데이터셋에서 평점 4~5 데이터
  - ***내용*** : Epinions 원본 데이터셋에서 실험에 필요한 형식으로 변환시킨 데이터



