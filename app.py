import streamlit as st
import twint
import pandas as pd
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Set page name and favicon
st.set_page_config(page_title='scraper & cluster',page_icon=':iphone:')
st.image('https://www.google.com/imgres?imgurl=https%3A%2F%2Fample.svce.edu.in%2Fwp-content%2Fuploads%2F2022%2F07%2FWeb-Mining.jpg&imgrefurl=https%3A%2F%2Fample.svce.edu.in%2Fweb-mining%2F&tbnid=3h7PLwkavcSk7M&vet=12ahUKEwjxjbbVkfn7AhXx9nMBHS6LB3sQMygEegUIARC5AQ..i&docid=ME_zePzS4XnfnM&w=440&h=220&q=web%20mining&ved=2ahUKEwjxjbbVkfn7AhXx9nMBHS6LB3sQMygEegUIARC5AQ')
st.subheader("""
Mari melakukan crawling data dengan menyenagkan!!!:
""")

# customize form
# with st.form(key='Twitter_form'):
    # search_term = st.text_input('Input data yang dicari')
    # limit = st.slider('Banyak tweet yg diinginkan', 0, 500, step=5)
    # output_csv = st.radio('Simpan file CSV?', ['Ya', 'Tdk'])
    # file_name = st.text_input('Nama file CSV:')
    # submit_button = st.form_submit_button(label='Cari')

    # if submit_button:
    #     # configure twint
    #     c = twint.Config()
    #     c.Search = search_term
    #     c.Limit = limit
    #     c.Store_csv = True

    #     if c.Store_csv:
    # c.Output = f'{file_name}.csv'
    #     twint.run.Search(c)

    #     st.markdown('hasil crawling data')
    #     data = pd.read_csv(f'{file_name}.csv', usecols=['date', 'tweet'])

data = pd.read_csv("https://raw.githubusercontent.com/sandiprayogo/ppw/main/Abstraksi.csv")
st.table(data)

st.subheader("Preprocessing Data")
st.markdown("""
proses untuk menyeleksi data text agar menjadi lebih terstruktur lagi dengan 
melalui serangkaian tahapan yang meliputi tahapan case folding, tokenizing, filtering dan stemming
""")

data['tweet'] = data['tweet'].str.lower()

def remove_special(text):
    # remove tab, new line, ans back slice
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\'," ").replace('\\f'," ").replace('\\r'," ")
    # remove non ASCII (emoticon, chinese word, .etc)
    text = text.encode('ascii', 'replace').decode('ascii')
    # remove mention, link, hashtag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    # remove incomplete URL
    return text.replace("http://", " ").replace("https://", " ")
data['tweet'] = data['tweet'].apply(remove_special)

#remove number
def remove_number(text):
    return  re.sub(r"\d+", "", text)
data['tweet'] = data['tweet'].apply(remove_number)

#remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans("","",string.punctuation))
data['tweet'] = data['tweet'].apply(remove_punctuation)

#remove whitespace leading & trailing
def remove_whitespace_LT(text):
    return text.strip()
data['tweet'] = data['tweet'].apply(remove_whitespace_LT)

#remove multiple whitespace into single whitespace
def remove_whitespace_multiple(text):
    return re.sub('\s+',' ',text)
data['tweet'] = data['tweet'].apply(remove_whitespace_multiple)

# remove single char
def remove_singl_char(text):
    return re.sub(r"\b[a-zA-Z]\b", " ", text)
data['tweet'] = data['tweet'].apply(remove_singl_char)

# token
nltk.download('punkt')
# NLTK word Tokenize 
def word_tokenize_wrapper(text):
    return word_tokenize(text)
data['tweet'] = data['tweet'].apply(word_tokenize_wrapper)

# filtering
nltk.download('stopwords')
list_stopwords = stopwords.words('indonesian')
# append additional stopword
list_stopwords.extend(["yg", "dg", "rt", "dgn", "ny", "d", 'klo', 
    'kalo', 'amp', 'biar', 'bikin', 'bilang', 
    'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 
    'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 
    'jd', 'jgn', 'sdh', 'aja', 'n', 't', 
    'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
    '&amp', 'yah'])
# convert list to dictionary
list_stopwords = set(list_stopwords)
#Menghapus Stopword dari list token
def stopwords_removal(words):
    return [word for word in words if word not in list_stopwords]
data['tweet'] = data['tweet'].apply(stopwords_removal)

# stemming
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter
# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()
# stemmed
def stemmed_wrapper(term):
    return stemmer.stem(term)
term_dict = {}

for document in data['tweet']:
    for term in document:
        if term not in term_dict:
            term_dict[term] = ' '
    
print(len(term_dict))
print("------------------------")

for term in term_dict:
    term_dict[term] = stemmed_wrapper(term)
    print(term,":" ,term_dict[term])
    
print(term_dict)
print("------------------------")

# apply stemmed term to dataframe
def get_stemmed_term(document):
    return [term_dict[term] for term in document]
data['tweet'] = data['tweet'].swifter.apply(get_stemmed_term)
data['tweet'].to_csv('Prepocessing.csv',index=False)
# st.table(data['tweet'])
st.dataframe(data['tweet'])

st.subheader("TF-IDF")
st.markdown("""
Algoritma TF-IDF (Term Frequency – Inverse Document Frequency) adalah salah satu algoritma yang dapat digunakan untuk menganalisa hubungan antara sebuah frase/kalimat dengan sekumpulan dokumen
""")
st.latex("$$tf-idf(t, d) = tf(t, d) * log(N/(df + 1))$$")
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#Membuat Dataframe
dataTextPre = pd.read_csv('Prepocessing.csv',index_col=False)
vectorizer = CountVectorizer(min_df=1)
bag = vectorizer.fit_transform(dataTextPre['tweet'])

"""### Matrik VSM(Visual Space Model)
Sebelum menghitung nilai TF, terlebih dahulu buat matrik vsm untuk menentukan bobot nilai term pada dokumen, hasilnya sebagaii berikut.
"""
matrik_vsm = bag.toarray()
matrik_vsm.shape
matrik_vsm[0]

a=vectorizer.get_feature_names()

dataTF =pd.DataFrame(data=matrik_vsm,index=list(range(1, len(matrik_vsm[:,1])+1, )),columns=[a])
dataTF.to_csv('TF.csv',index=False)
st.dataframe(dataTF)

"""### Nilai Term Dokumen
Setelah didapat nilai matrik vsm, maka nilai term frequency yang didapat pada masing masing dokumen ialah seperti berikut.
"""
datalabel = pd.read_csv('Prepocessing.csv',index_col=False)
TF = pd.read_csv('TF.csv',index_col=False)
dataJurnal = pd.concat([TF, datalabel["tweet"]], axis=1)
st.dataframe(dataJurnal)

st.subheader("Kmeans")
st.markdown("""
K-Means Clustering adalah suatu metode penganalisaan data atau metode Data Mining yang melakukan proses pemodelan unssupervised learning dan menggunakan metode yang mengelompokan data berbagai partisi.
K Means Clustering memiliki objective yaitu meminimalisasi object function yang telah di atur pada proses clasterisasi. Dengan cara minimalisasi variasi antar 1 cluster dengan maksimalisasi variasi dengan data di cluster lainnya.
K means clustering merupakan metode algoritma dasar,yang diterapkan sebagai berikut: 
Menentukan jumlah cluster
###a. Secara acak mendistribusikan data cluster
###b. Menghitung rata rata dari data yang ada di cluster.
###c. Menggunakan langkah baris 3 kembali sesuai nilai treshold
###d. Menghitung jarak antara data dan nilai centroid(K means clustering)
###e. Distance space dapat diimplementasikan untuk menghitung jarak data dan centroid. Contoh penghitungan jarak yang sering digunakan adalah manhattan/city blok distance
""")
st.latex("$$d(p,q) = \sqrt{(p_{1}-q_{1})^2+(p_{2}-q_{2})^2+(p_{3}-q_{3})^2}$$")

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
# Latih Kmeans dengan n cluster terbaik
modelKm = KMeans(n_clusters=2, random_state=12)
modelKm.fit(dataTF.values)
prediksi = modelKm.predict(dataTF.values)

# Pengurangan dimensi digunakan untuk memplot dalam representasi 2d
pc=TruncatedSVD(n_components=2)
X_new=pc.fit_transform(dataTF.values)
centroids=pc.transform(modelKm.cluster_centers_)
centroids

dataTF['Cluster_Id'] = modelKm.labels_
dataTF

fig, ax = plt.subplots()
ax.scatter(X_new[:,0],X_new[:,1],c=prediksi, cmap='viridis')
ax.scatter(centroids[:,0] , centroids[:,1] , s = 50, color = 'red')
    
plt.tight_layout()
st.write(fig)