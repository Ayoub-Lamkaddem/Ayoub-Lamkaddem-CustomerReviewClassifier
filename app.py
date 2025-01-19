import streamlit as st
import joblib
import os
import torch
from transformers import AutoTokenizer, BertForSequenceClassification, pipeline, XLNetForSequenceClassification, XLNetTokenizer

# Add custom page configuration
st.set_page_config(
    page_title="Sentiment Analysis App 📊", 
    page_icon="🔍", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved styling
st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        color: #2C3E50;
    }
    .stButton>button {
        background-color: #3498DB;
        color: white;
        border-radius: 10px;
    }
    .stTextArea>div>div>textarea {
        border: 2px solid #3498DB;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Fonction pour charger le modèle et les dépendances
def load_model(model_name):
    try:
        if "SVM" in model_name:
            # Load SVM model
            model = joblib.load('SVM/svm_model.pkl')
            vectorizer = joblib.load('SVM/tfidf_vectorizer.pkl')
            # No label encoder needed for SVM as mentioned
            return model, vectorizer, None
        elif "Régression" in model_name:
            # Existing Logistic Regression model loading
            model = joblib.load('Logistic Regression/logreg_model.pkl')
            vectorizer = joblib.load('Logistic Regression/tfidf_vectorizer.pkl')
            label_encoder = joblib.load('Logistic Regression/label_encoder.pkl')
            return model, vectorizer, label_encoder
        elif "BERT" in model_name:
            try:
                # Chemin du modèle BERT
                path = "fine_tuned_BERT"
                
                # Vérifier l'existence du dossier
                if not os.path.exists(path):
                    raise ValueError(f"Le dossier {path} n'existe pas")
                
                # Charger le modèle et le tokenizer
                model = BertForSequenceClassification.from_pretrained(path, num_labels=3)
                tokenizer = AutoTokenizer.from_pretrained(path)
                
                # Mettre le modèle en mode évaluation
                model.eval()
                
                return model, tokenizer, None
            
            except Exception as e:
                st.error(f"Erreur lors du chargement du modèle BERT : {str(e)}")
                # Afficher le contenu du dossier pour le débogage
                st.write("Contenu du dossier fine_tuned_BERT:")
                st.write(os.listdir("fine_tuned_BERT"))
                return None, None, None
        elif "XLNET" in model_name:
            try:
                # Chemin du modèle XLNet
                path = "fine_tuned_XLNET"
                
                # Vérifier l'existence du dossier
                if not os.path.exists(path):
                    raise ValueError(f"Le dossier {path} n'existe pas")
                
                # Charger le modèle et le tokenizer
                model = XLNetForSequenceClassification.from_pretrained(path, num_labels=3)
                tokenizer = XLNetTokenizer.from_pretrained(path)
                
                # Mettre le modèle en mode évaluation
                model.eval()
                
                return model, tokenizer, None
            
            except Exception as e:
                st.error(f"Erreur lors du chargement du modèle XLNet : {str(e)}")
                # Afficher le contenu du dossier pour le débogage
                st.write("Contenu du dossier fine_tuned_XLNET:")
                st.write(os.listdir("fine_tuned_XLNET"))
                return None, None, None
        else:
            st.error(f"Modèle non pris en charge : {model_name}")
            return None, None, None
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None, None, None

# Fonction pour effectuer la prédiction
def predict_sentiment(text, model, vectorizer, label_encoder):
    if not text.strip():
        return "Erreur : Le champ de texte est vide. "
    try:
        # Gestion spécifique pour le modèle BERT et XLNet
        if isinstance(model, (BertForSequenceClassification, XLNetForSequenceClassification)):
            # Tokeniser et préparer l'entrée pour BERT/XLNet
            inputs = vectorizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Faire la prédiction
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.softmax(outputs.logits, dim=1)
                predicted_class = torch.argmax(predictions, dim=1).item()
            
            # Mapper les classes
            sentiment_labels = ['negative', 'neutral', 'positive']
            predicted_label = sentiment_labels[predicted_class]
            
            # Mapping des sentiments
            sentiment_map = {
                "positive": ("🌟 Commentaire positif ! 🎉 Le commentaire est très enthousiaste et encourageant 🚀", "success"),
                "negative": ("😔 Sentiment négatif ! 🚫 Le commentaire exprime de la frustration ou de l'insatisfaction 💥", "error"), 
                "neutral": ("😐 Commentaire neutre 🤷‍♀️ Aucune émotion forte n'est exprimée ", "warning")
            }
            
            message, color_type = sentiment_map[predicted_label]
            
            # Utiliser la méthode de message coloré appropriée
            if color_type == "success":
                st.success(message)
            elif color_type == "error":
                st.error(message)
            elif color_type == "warning":
                st.warning(message)
            
            return message
        
        # Code existant pour SVM et Régression Logistique
        # Vectorisation du texte
        text_tfidf = vectorizer.transform([text])
        # Prédiction
        predicted_rating = model.predict(text_tfidf)
        
        # Décodage de la prédiction
        if label_encoder:
            predicted_rating_label = label_encoder.inverse_transform(predicted_rating)
        else:
            # Pour SVM, essayons de convertir directement
            if predicted_rating[0] in [0, 1, 2]:
                predicted_rating_label = ['negative', 'neutral', 'positive'][predicted_rating[0]]
            else:
                predicted_rating_label = [str(predicted_rating[0])]
        
        # Mapping flexible pour gérer différents formats de labels
        sentiment_map = {
            "positive": ("🌟 Commentaire positif ! 🎉 Le commentaire est très enthousiaste et encourageant 🚀", "success"),
            "negative": ("😔 Sentiment négatif ! 🚫 Le commentaire exprime de la frustration ou de l'insatisfaction 💥", "error"), 
            "neutral": ("😐 Commentaire neutre 🤷‍♀️ Aucune émotion forte n'est exprimée ", "warning")
        }
        
        # Convertir en minuscules pour correspondance insensible à la casse
        label = str(predicted_rating_label[0]).lower()
        
        # Vérifier si le label existe dans le mapping
        if label in sentiment_map:
            message, color_type = sentiment_map[label]
            
            # Utiliser la méthode de message coloré appropriée
            if color_type == "success":
                st.success(message)
            elif color_type == "error":
                st.error(message)
            elif color_type == "warning":
                st.warning(message)
            
            return message
        else:
            # Si le label n'est pas dans le mapping, afficher un message générique
            st.info(f"Sentiment détecté : {label}")
            return f"Sentiment détecté : {label}"
    except Exception as e:
        return f"Erreur lors de la prédiction : {e}"

# Interface Streamlit
st.title("🌟 Analyse de Sentiment avec Différents Modèles")

# Barre latérale
st.sidebar.header("🛠️ Paramètres")

# Sélection du modèle
model_options = ["Régression Logistique 📈", "SVM 🤖", "BERT 🧠", "XLNET 🔮"]
selected_model = st.sidebar.selectbox("Sélectionnez un modèle :", model_options)

# Chargement du modèle
if st.sidebar.button("🔬 Charger le modèle"):
    model, vectorizer, label_encoder = load_model(selected_model)
    if model and vectorizer and label_encoder:
        st.sidebar.success(f"Modèle {selected_model} chargé avec succès ! ✅")
        st.session_state.model = model
        st.session_state.vectorizer = vectorizer
        st.session_state.label_encoder = label_encoder
    elif model and vectorizer:
        st.sidebar.success(f"Modèle {selected_model} chargé avec succès ! ")
        st.session_state.model = model
        st.session_state.vectorizer = vectorizer
        st.session_state.label_encoder = None
    else:
        st.sidebar.error("❌ Échec du chargement du modèle.")

# Zone principale
st.header("🔍 Analyse de Sentiment  📊")
st.markdown('<p class="big-font">Cette application permet de prédire si un commentaire est <b>positif</b>, <b>négatif</b>, ou <b>neutre</b> en fonction du modèle sélectionné. </p>', unsafe_allow_html=True)

# Message initial
st.info("👋 Bienvenue ! Commencez par charger un modèle dans la barre latérale, puis entrez un commentaire pour analyser son sentiment. 🕵️‍♀️")
# Champ de saisie pour le commentaire
user_input = st.text_area("📝 Entrez un commentaire :", "")

# Bouton de prédiction
if st.button("🚀 Prédire le sentiment"):
    if "model" in st.session_state and "vectorizer" in st.session_state:
        result = predict_sentiment(user_input, st.session_state.model, st.session_state.vectorizer, st.session_state.label_encoder)
    else:
        st.warning("⚠️ Veuillez d'abord charger un modèle dans la barre latérale.")