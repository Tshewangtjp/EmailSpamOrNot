import streamlit as st
import pickle
import nltk
import string

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="Spam Classifier",
    page_icon="📧",
    layout="centered"
)

# -----------------------
# Custom CSS
# -----------------------
st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

.title{
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#4F46E5;
}

.subtitle{
    text-align:center;
    color:gray;
    margin-bottom:25px;
}

.stButton>button{
    width:100%;
    background:linear-gradient(90deg,#4F46E5,#7C3AED);
    color:white;
    border:none;
    border-radius:12px;
    padding:12px;
    font-size:18px;
    font-weight:bold;
}

.stButton>button:hover{
    background:linear-gradient(90deg,#4338CA,#6D28D9);
}

.result-box{
    padding:20px;
    border-radius:15px;
    text-align:center;
    font-size:22px;
    font-weight:bold;
}

.spam{
    background:#FEE2E2;
    color:#B91C1C;
}

.ham{
    background:#DCFCE7;
    color:#15803D;
}

textarea{
    border-radius:12px !important;
}

</style>
""", unsafe_allow_html=True)

# -----------------------
# Preprocessing
# -----------------------
def transform_text(text):

    text = text.lower()

    words = word_tokenize(text)

    y=[]

    for word in words:
        if word.isalnum():
            y.append(word)

    words=y[:]
    y.clear()

    for word in words:
        if word not in stopwords.words('english') and word not in string.punctuation:
            y.append(word)

    words=y[:]
    y.clear()

    for word in words:
        y.append(ps.stem(word))

    return " ".join(y)

# -----------------------
# Load Model
# -----------------------
tfidf = pickle.load(open("vectorizer.pkl","rb"))
model = pickle.load(open("model.pkl","rb"))

# -----------------------
# Header
# -----------------------
st.markdown("<div class='title'>📧 Email Spam Detector</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>AI-powered spam message classification</div>",
    unsafe_allow_html=True
)

# -----------------------
# Input
# -----------------------
message = st.text_area(
    "✍ Enter your message",
    height=180,
    placeholder="Type or paste your Email here..."
)

col1,col2 = st.columns(2)

with col1:
    st.metric("Characters", len(message))

with col2:
    st.metric("Words", len(message.split()))

# -----------------------
# Example
# -----------------------
with st.expander("💡 Example Spam Message"):
    st.write(
        "Congratulations! You have won ₹50,000. Click here to claim your prize now!"
    )

# -----------------------
# Prediction
# -----------------------
if st.button("🔍 Analyze Message"):

    if message.strip()=="":

        st.warning("Please enter a message.")

    else:

        with st.spinner("Analyzing..."):

            transformed = transform_text(message)

            vector = tfidf.transform([transformed])

            prediction = model.predict(vector)[0]

            if hasattr(model,"predict_proba"):
                confidence = model.predict_proba(vector).max()*100
            else:
                confidence = None

        st.divider()

        if prediction==1:

            st.markdown("""
            <div class='result-box spam'>
            🚨 SPAM MESSAGE
            </div>
            """,unsafe_allow_html=True)

        else:

            st.markdown("""
            <div class='result-box ham'>
            ✅ SAFE MESSAGE
            </div>
            """,unsafe_allow_html=True)

        if confidence:
            st.progress(confidence/100)
            st.write(f"**Confidence:** {confidence:.2f}%")