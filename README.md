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
  
  '''
      [ Hyperparameter Tuning ]
      # Before you start ... 
       - Empty list can not be used for hyperparameter
       - Renew directory @main.py(line 6)
      
      # Hyperparameter infos
       - models : {'GMF' | 'MLP' | 'NeuMF'}
       - epochs : set training epochs numbers
       - NlatentUsers : set latent user numbers if models == {'MLP' | 'NeuMF'}
       - NlatentItems : set latent item numbers if models == {'MLP' | 'NeuMF'}
       - NlatentMFs : set latent matrix factorization numbers if models == {'GMF' | 'NeuMF'}
       - term : set OCF-B iteration number
       - user_ids : set range of user data id
       - item_ids : set range of item data id
       - neg_cases_mov : set number of negative case data to generate
       - graph_list : set metrics which you want to print training history
      '''
      
  if __name__ == "__main__":
      
      models = ['GMF']
      epochs = [10,15]
      NlatentUsers = [1,2,3]
      NlatentItems = [1,2,3] 
      NlatentMFs = [3,4]
      terms = 10
      user_ids = [1, 943]
      item_ids = [1, 1682]
      neg_cases_mov = [25000, 50000, 75000]
  
      graph_list = [
        	"AUC", "RMSE", 
        	"Positive_RMSE", "Negative_RMSE", 
        	"Positive_Precision", "Negative_Precision", 
        	"Positive_Recall", "Negative_Recall"]
  
      experiment_by_model(
        models, epochs, NlatentUsers, NlatentItems, NlatentMFs, 
        neg_cases_mov, terms, "MovieLens100K_oneClass.csv", 
        user_ids, item_ids, graph_list)
  ```

  
