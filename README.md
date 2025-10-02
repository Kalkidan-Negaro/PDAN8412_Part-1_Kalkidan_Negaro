# Blog Authorship Analysis with LSTM

## Project Description

This project **analyses** a large dataset of blog posts to predict authorship using Long Short-Term Memory (LSTM) neural networks. It shows text **preprocessing**, exploratory data analysis, model training, and evaluation using Python, TensorFlow, and PySpark.

## Dataset

  - **Name:** Blog Authorship Corpus  
  - **Source:** [Kaggle](https://www.kaggle.com/rtatman/blog-authorship-corpus)  
  - **Records:** 681,000  
  - **Features:**  
      - `id`: Author ID  
      - `text`: Blog post content  
      - Metadata: `date`, `gender`, `age`, `topic`, `sign`  

## Tools and Libraries

  - **Python Version:** 3.12  
  - **Libraries:**  
      - PySpark (for scalable data manipulation)  
      - Pandas (data handling)  
      - NumPy (numerical operations)  
      - Matplotlib (**visualisation**)  
      - Seaborn (**visualisation**)  
      - TensorFlow/Keras (LSTM model)

## Steps

1.  **Data Loading:** Load the CSV file into Spark DataFrame.  
2.  **Exploratory Data Analysis:**  
       - Check schema, null values, and distributions  
       - **Analyse** post length  
       - **Visualise** top authors and word clouds  
3.  **Preprocessing for LSTM:**  
       - Sample the dataset  
       - Encode authors  
       - Tokenize and pad text  
       - Split into training and test sets  
4.  **Model Training:** Train LSTM with embedding, dropout, and softmax output.  
5.  **Evaluation:**  
       - Generate classification report  
       - Plot confusion matrix for top authors  
6.  **Reporting:** **Summarise** findings, model accuracy, and insights.

## How to Run

1.  **Clone the Repository:**
       \`\`\`bash
       git clone https://github.com/VCDN-2025/pdan8412-part-1-Kalkidan-Negaro.git
       cd pdan8412-part-1-Kalkidan-Negaro

2.  ## Install Dependencies

<!-- end list -->

```bash
pip install -r requirements.txt


## Run Notebook in Google Colab or Jupyter Notebook

1. Open `BlogAuthorship.ipynb`.
2. Execute cells sequentially to reproduce analysis and model training.

## References

- **Blog Authorship Corpus Dataset:** [https://www.kaggle.com/rtatman/blog-authorship-corpus](https://www.kaggle.com/rtatman/blog-authorship-corpus)  
- **TensorFlow Documentation:** [https://www.tensorflow.org/](https://www.tensorflow.org/)  
- **Keras LSTM Guide:** [https://keras.io/api/layers/recurrent_layers/lstm/](https://keras.io/api/layers/recurrent_layers/lstm/)  
- **PySpark Documentation:** [https://spark.apache.org/docs/latest/api/python/](https://spark.apache.org/docs/latest/api/python/)
```
