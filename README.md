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
      '''
      [ Hyperparameter Tuning ]
      * 주의 : empty list가 있으면 안됨
      * 주의 : 경로 재설정 필요 (line 6)
      '''
      models = ['GMF']       # {'GMF' | 'MLP' | 'NeuMF'}
      epochs = [10,15]
      NlatentUsers = [1,2,3] # 해보고 싶은 latent user --> {'MLP' | 'NeuMF'}의 경우
      NlatentItems = [1,2,3] # 해보고 싶은 latent item --> {'MLP' | 'NeuMF'}의 경우
      NlatentMFs = [3,4]   	 # 해보고 싶은 latent mf   --> {'GMF' | 'NeuMF'}의 경우
      terms = 10
  
      # 프린트하고 싶은 Metric들
      graph_list = [
        	"AUC", "RMSE", 
        	"Positive_RMSE", "Negative_RMSE", 
        	"Positive_Precision", "Negative_Precision", 
        	"Positive_Recall", "Negative_Recall"]
  
      neg_cases_mov = [25000, 50000, 75000]
      experiment_by_model(
        models, epochs, NlatentUsers, NlatentItems, NlatentMFs, 
        neg_cases_mov, terms, "MovieLens100K_oneClass.csv", 
        [1, 943], [1, 1682], graph_list)
  ```

  
