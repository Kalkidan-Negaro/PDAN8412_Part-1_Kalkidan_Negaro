# Blog Authorship Analysis with LSTM ✍️

This repository contains the code and resources for a project focused on predicting blog post authorship using advanced Natural Language Processing (NLP) techniques, specifically **Long Short-Term Memory (LSTM)** neural networks. The analysis is performed on a large corpus of blog data, leveraging PySpark for scalable data manipulation.

-----

## Project Description

This project **analyses** a large dataset of blog posts to predict authorship using Long Short-Term Memory (LSTM) neural networks. It covers essential steps in a typical data science workflow: text **preprocessing**, **exploratory data analysis** (EDA), model training, and **evaluation** using a suite of modern tools including Python, TensorFlow, and PySpark.

-----

## Dataset

The core of this project is the extensive **Blog Authorship Corpus**.

| Detail | Value |
| :--- | :--- |
| **Name** | Blog Authorship Corpus |
| **Source** | [Kaggle](https://www.kaggle.com/rtatman/blog-authorship-corpus) |
| **Records** | 681,000 |
| **Features** | `id` (Author ID), `text` (Blog post content), Metadata (`date`, `gender`, `age`, `topic`, `sign`) |

-----

## Tools and Libraries 🛠️

The following versions and libraries are required to run this project:

| Category | Tool/Library | Purpose |
| :--- | :--- | :--- |
| **Environment** | **Python** 3.12 | Core programming language |
| **Scalability** | **PySpark** | Scalable data manipulation and loading |
| **Data Handling** | **Pandas**, **NumPy** | Data handling and numerical operations |
| **Visualisation** | **Matplotlib**, **Seaborn** | Data and results **visualisation** |
| **Machine Learning** | **TensorFlow/Keras** | Building and training the **LSTM** model |

-----

## Methodology and Steps

The project follows a six-step methodology to achieve the authorship prediction goal:

1.  **Data Loading:** Load the CSV file into a Spark DataFrame for efficient, large-scale processing.

2.  **Exploratory Data Analysis (EDA):**

      * Check schema, null values, and feature distributions.
      * **Analyse** post length distributions across authors.
      * **Visualise** top authors and generate word clouds to understand corpus vocabulary.

3.  **Preprocessing for LSTM:**

      * Sample the dataset to manage memory and training time (due to the size).
      * Encode authors (the target variable) numerically.
      * Tokenize and pad text sequences to create fixed-length inputs suitable for the LSTM.
      * Split the processed data into training and test sets.

4.  **Model Training:** Train an **LSTM** sequential model incorporating:

      * An **Embedding layer** to represent words as dense vectors.
      * **Dropout** for regularisation.
      * A **Softmax** output layer for multi-class classification (author prediction).

5.  **Evaluation:**

      * Generate a detailed **classification report** (precision, recall, F1-score).
      * Plot a **confusion matrix** for the top authors to understand misclassification patterns.

6.  **Reporting:** Summarise key findings, model **accuracy**, and practical insights derived from the authorial style analysis.

-----

## How to Run 🚀

To reproduce this analysis and model training, follow the steps below:

### 1\. Clone the Repository

Open your terminal and clone the project files:

```bash
git clone https://github.com/VCDN-2025/pdan8412-part-1-Kalkidan-Negaro.git
cd pdan8412-part-1-Kalkidan-Negaro
```

### 2\. Install Dependencies

Install all necessary Python libraries using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 3\. Run the Notebook

Open the main notebook file in your preferred environment:

1.  Open `BlogAuthorship.ipynb` in **Google Colab** or a **Jupyter Notebook** instance.
2.  Execute the cells sequentially to run the entire analysis pipeline, from data loading to model evaluation.

-----

## References

  * **Blog Authorship Corpus Dataset:** [https://www.kaggle.com/rtatman/blog-authorship-corpus](https://www.kaggle.com/rtatman/blog-authorship-corpus)
  * **TensorFlow Documentation:** [https://www.tensorflow.org/](https://www.tensorflow.org/)
  * **Keras LSTM Guide:** [https://keras.io/api/layers/recurrent\_layers/lstm/](https://keras.io/api/layers/recurrent_layers/lstm/)
  * **PySpark Documentation:** [https://spark.apache.org/docs/latest/api/python/](https://spark.apache.org/docs/latest/api/python/)
