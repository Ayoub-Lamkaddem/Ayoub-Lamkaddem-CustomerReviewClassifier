# 📊 Customer Reviews Classification: A Comparison Between Traditional Machine Learning and Transfer Learning

This project explores the performance of traditional machine learning techniques such as Support Vector Machines (SVM) and Logistic Regression versus modern transfer learning methods like BERT and XLNet for customer review classification.

## 📝 Overview
The aim is to determine which approach offers better accuracy and efficiency in classifying customer reviews into categories such as positive, negative, or neutral.

## ✨ Features
- **🛠️ Data Preprocessing**: 
  - Text cleaning (removing stop words, punctuation, etc.).
  - Tokenization and vectorization.
- **⚙️ Traditional Machine Learning**: 
  - Implementation of SVM and Logistic Regression models using TF-IDF features.
- **🚀 Transfer Learning**: 
  - Fine-tuning pre-trained BERT and XLNet models.
- **📈 Evaluation**: 
  - Comparison of accuracy, precision, recall, and F1-score between the approaches.

## 🛠️ Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - 🐍 scikit-learn
  - 🐼 pandas
  - 📊 numpy
  - 🤗 transformers (Hugging Face)
  - 🔥 PyTorch
  - 📝 NLTK/Spacy for NLP preprocessing

## 📂 Project Structure
```
CustomerReviewClassifier/
├── Logistic Regression/
│   ├── Logistic_Regression.ipynb
│   ├── label_encoder.pkl
│   ├── logreg_model.pkl
│   ├── tfidf_vectorizer.pkl
├── SVM/
│   ├── SVM.ipynb
│   ├── svm_model.pkl
│   ├── tfidf_vectorizer.pkl
├── fine_tuned_BERT/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── vocab.txt
├── xlnet_model/
│   ├── config.json
│   ├── model.safetensors
│   ├── special_tokens_map.json
│   ├── spiece.model
│   ├── tokenizer_config.json
├── BERT_lora.ipynb
├── Data_preprocessing.ipynb
├── Rapport ML.pdf
├── Xlnet&Comparaison.ipynb
├── app.py
├── balanced_subset.csv
├── requirements.txt           # List of dependencies
├── README.md                  # Project overview and instructions
└── LICENSE                    # License information
```

## ⚡ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/AyoubLamkaddem/CustomerReviewClassifier.git
   cd CustomerReviewClassifier
   ```

## 🚀 Usage
1. Prepare the dataset:
   - Ensure `balanced_subset.csv` is in the project root.
2. Run the data preprocessing notebook:
   ```bash
   jupyter notebook Data_preprocessing.ipynb
   ```
3. Train the models:
   - Logistic Regression: Run `Logistic_Regression.ipynb` in the `Logistic Regression/` folder.
   - SVM: Run `SVM.ipynb` in the `SVM/` folder.
   - Fine-tune BERT: Use the files in the `fine_tuned_BERT/` directory.
   - XLNet: Use the files in the `xlnet_model/` directory and `Xlnet&Comparaison.ipynb`.
4. Deploy the application:
   ```bash
   python app.py
   ```

## 📊 Results
The project provides a detailed comparison of traditional ML and transfer learning models, including metrics like accuracy, precision, recall, and F1-score. Results are documented in `Rapport ML.pdf`.

## 🔮 Future Work
- Experiment with other transfer learning models like RoBERTa or GPT.
- Optimize hyperparameters for both traditional ML and transfer learning models.
- Extend to multilingual datasets.

## 🤝 Contributing
Feel free to fork the repository and submit pull requests. Contributions are welcome!

## 📜 License
This project is licensed under the MIT License. See the LICENSE file for details.
