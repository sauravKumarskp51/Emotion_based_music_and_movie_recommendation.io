from __future__ import division, print_function
#import sys
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras
import cv2
#import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import statistics as st


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index1.html")


@app.route('/camera', methods = ['GET', 'POST'])
def camera():
    i=0

    GR_dict={0:(0,255,0),1:(0,0,255)}
    model = tf.keras.models.load_model('final_model.h5')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    output=[]
    cap = cv2.VideoCapture(0)
    while (i<=30):
        ret, img = cap.read()
        faces = face_cascade.detectMultiScale(img,1.05,5)

        for x,y,w,h in faces:

            face_img = img[y:y+h,x:x+w]

            resized = cv2.resize(face_img,(224,224))
            reshaped=resized.reshape(1, 224,224,3)/255
            predictions = model.predict(reshaped)

            max_index = np.argmax(predictions[0])

            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'neutral', 'surprise')
            predicted_emotion = emotions[max_index]
            output.append(predicted_emotion)



            cv2.rectangle(img,(x,y),(x+w,y+h),GR_dict[1],2)
            cv2.rectangle(img,(x,y-40),(x+w,y),GR_dict[1],-1)
            cv2.putText(img, predicted_emotion, (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        i = i+1

        cv2.imshow('LIVE', img)
        key = cv2.waitKey(1)
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            break
    print(output)
    cap.release()
    cv2.destroyAllWindows()
    final_output1 = st.mode(output)
    return render_template("buttons.html",final_output=final_output1)


@app.route('/templates/buttons', methods = ['GET','POST'])
def buttons():
    return render_template("buttons.html")

@app.route('/movies/surprise', methods = ['GET', 'POST'])
def moviesSurprise():
    return render_template("moviesSurprise.html")

@app.route('/movies/angry', methods = ['GET', 'POST'])
def moviesAngry():
    return render_template("moviesAngry.html")

@app.route('/movies/sad', methods = ['GET', 'POST'])
def moviesSad():
    return render_template("moviesSad.html")

@app.route('/movies/disgust', methods = ['GET', 'POST'])
def moviesDisgust():
    return render_template("moviesDisgust.html")

@app.route('/movies/happy', methods = ['GET', 'POST'])
def moviesHappy():
    return render_template("moviesHappy.html")

@app.route('/movies/fear', methods = ['GET', 'POST'])
def moviesFear():
    return render_template("moviesFear.html")

@app.route('/movies/neutral', methods = ['GET', 'POST'])
def moviesNeutral():
    return render_template("moviesNeutral.html")

@app.route('/songs/surprise', methods = ['GET', 'POST'])
def songsSurprise():
    return render_template("songsSurprise.html")

@app.route('/songs/angry', methods = ['GET', 'POST'])
def songsAngry():
    return render_template("songsAngry.html")

@app.route('/songs/sad', methods = ['GET', 'POST'])
def songsSad():
    return render_template("songsSad.html")

@app.route('/songs/disgust', methods = ['GET', 'POST'])
def songsDisgust():
    return render_template("songsDisgust.html")

@app.route('/songs/happy', methods = ['GET', 'POST'])
def songsHappy():
    return render_template("songsHappy.html")

@app.route('/songs/fear', methods = ['GET', 'POST'])
def songsFear():
    return render_template("songsFear.html")

@app.route('/songs/neutral', methods = ['GET', 'POST'])
def songsNeutral():
    return render_template("songsSad.html")

@app.route('/templates/join_page', methods = ['GET', 'POST'])
def join():
    return render_template("join_page.html")
    
if __name__ == "__main__":
    app.run(debug=True)
    
    
    http://localhost:5000/camera
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    @app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data['email_id']
    password = data['password']

    # Fetch user from the database
    user = User.query.filter_by(email_id=email).first()

    if user and check_password_hash(user.password, password):
        return render_template("index1.html")

    return jsonify(success=False, message='Invalid credentials')

    
    
    
    
    
    
    if (speechResult.includes('play')) {
                    var songName = speechResult.replace('play', '').trim();
                    speak('Playing ' + songName + '', function() {
                        playSpotifySong(songName);
                    });
                }
    
   
    
    
    
    
    background-image: linear-gradient(rgba(22, 22, 21, 0.5), rgba(10, 10, 10, 0.849)), url(static/img12.png);
    
    























































// Hindi Horror Movies
var hindiHorrorMovies = [
    { title: "1920", language: "Hindi", imdbUrl: "https://www.imdb.com/title/tt1301698/" },
    { title: "Pari", language: "Hindi", imdbUrl: "https://www.imdb.com/title/tt7329858/" },
    { title: "Stree", language: "Hindi", imdbUrl: "https://www.imdb.com/title/tt8108202/" },
    { title: "Raat", language: "Hindi", imdbUrl: "https://www.imdb.com/title/tt0102558/" },
    { title: "Ragini MMS", language: "Hindi", imdbUrl: "https://www.imdb.com/title/tt1705772/" },
    { title: "Bhoot", language: "Hindi", imdbUrl: "https://www.imdb.com/title/tt0341711/" },
    { title: "Raaz", language: "Hindi", imdbUrl: "https://www.imdb.com/title/tt0299078/" },
    { title: "Veerana", language: "Hindi", imdbUrl: "https://www.imdb.com/title/tt0288083/" },
    { title: "Raaz 3D", language: "Hindi", imdbUrl: "https://www.imdb.com/title/tt1926313/" },
    { title: "Mahal", language: "Hindi", imdbUrl: "https://www.imdb.com/title/tt0047146/" },
    { title: "Darna Zaroori Hai", language: "Hindi", imdbUrl: "https://www.imdb.com/title/tt0471571/" },
    { title: "Darna Mana Hai", language: "Hindi", imdbUrl: "https://www.imdb.com/title/tt0341304/" },
    { title: "Ragini MMS 2", language: "Hindi", imdbUrl: "https://www.imdb.com/title/tt2609218/" },
    { title: "13B: Fear Has a New Address", language: "Hindi", imdbUrl: "https://www.imdb.com/title/tt1337083/" },
    { title: "1920: Evil Returns", language: "Hindi", imdbUrl: "https://www.imdb.com/title/tt2222550/" },
    { title: "Horror Story", language: "Hindi", imdbUrl: "https://www.imdb.com/title/tt2991526/" },
    { title: "Tumbbad", language: "Hindi", imdbUrl: "https://www.imdb.com/title/tt8239946/" },
    { title: "Raaz Reboot", language: "Hindi", imdbUrl: "https://www.imdb.com/title/tt5639388/" },
    { title: "Aatma", language: "Hindi", imdbUrl: "https://www.imdb.com/title/tt2375559/" },
    { title: "Ek Thi Daayan", language: "Hindi", imdbUrl: "https://www.imdb.com/title/tt2229842/" },
    { title: "The House Next Door", language: "Hindi", imdbUrl: "https://www.imdb.com/title/tt7102458/" },
    { title: "1921", language: "Hindi", imdbUrl: "https://www.imdb.com/title/tt7218518/" },
    { title: "Darna Zaroori Hai", language: "Hindi", imdbUrl: "https://www.imdb.com/title/tt0471571/" },
    { title: "Shaapit: The Cursed", language: "Hindi", imdbUrl: "https://www.imdb.com/title/tt1434517/" },
    { title: "Ek Paheli Leela", language: "Hindi", imdbUrl: "https://www.imdb.com/title/tt4500734/" },
    { title: "Veerana", language: "Hindi", imdbUrl: "https://www.imdb.com/title/tt0288083/" },
    { title: "Raaz: The Mystery Continues", language: "Hindi", imdbUrl: "https://www.imdb.com/title/tt1170413/" },
    { title: "Haunted - 3D", language: "Hindi", imdbUrl: "https://www.imdb.com/title/tt1781837/" },
    { title: "Dobaara: See Your Evil", language: "Hindi", imdbUrl: "https://www.imdb.com/title/tt5214948/" },
    { title: "Kaalo", language: "Hindi", imdbUrl: "https://www.imdb.com/title/tt1612578/" }
];




// Kannada Horror Movies
var kannadaHorrorMovies = [
    { title: "Shhh!", language: "Kannada", imdbUrl: "https://www.imdb.com/title/tt0360031/" },
    { title: "Aake", language: "Kannada", imdbUrl: "https://www.imdb.com/title/tt6899820/" },
    { title: "Karva", language: "Kannada", imdbUrl: "https://www.imdb.com/title/tt6184096/" },
    { title: "Chandralekha", language: "Kannada", imdbUrl: "https://www.imdb.com/title/tt1220278/" },
    { title: "Kanchana Ganga", language: "Kannada", imdbUrl: "https://www.imdb.com/title/tt2122270/" },
    { title: "Aatagara", language: "Kannada", imdbUrl: "https://www.imdb.com/title/tt4484576/" },
    { title: "Ee Bhoomi Aa Bhoomi", language: "Kannada", imdbUrl: "https://www.imdb.com/title/tt2198658/" },
    { title: "Mantram", language: "Kannada", imdbUrl: "https://www.imdb.com/title/tt1287834/" },
    { title: "Rangamandira", language: "Kannada", imdbUrl: "https://www.imdb.com/title/tt0213671/" },
    { title: "Bayalu Daari", language: "Kannada", imdbUrl: "https://www.imdb.com/title/tt1625021/" },
    { title: "Chandrika", language: "Kannada", imdbUrl: "https://www.imdb.com/title/tt4988790/" },
    { title: "Mantra", language: "Kannada", imdbUrl: "https://www.imdb.com/title/tt1365429/" },
    { title: "Kathe Chitrakathe Nirdeshana Puttanna", language: "Kannada", imdbUrl: "https://www.imdb.com/title/tt4460670/" },
    { title: "Alone", language: "Kannada", imdbUrl: "https://www.imdb.com/title/tt5813808/" },
    { title: "Apthamitra", language: "Kannada", imdbUrl: "https://www.imdb.com/title/tt0374202/" },
    { title: "Apoorva", language: "Kannada", imdbUrl: "https://www.imdb.com/title/tt0358028/" },
    { title: "Mummy - Save Me", language: "Kannada", imdbUrl: "https://www.imdb.com/title/tt6272364/" },
    { title: "Neenyare", language: "Kannada", imdbUrl: "https://www.imdb.com/title/tt0409317/" },
    { title: "Ouija", language: "Kannada", imdbUrl: "https://www.imdb.com/title/tt4844570/" },
    { title: "Deadly 2", language: "Kannada", imdbUrl: "https://www.imdb.com/title/tt5899884/" },
    { title: "Fear", language: "Kannada", imdbUrl: "https://www.imdb.com/title/tt0366298/" },
    { title: "Khaalida Nee Kate", language: "Kannada", imdbUrl: "https://www.imdb.com/title/tt3696740/" },
    { title: "Kshana Kshana", language: "Kannada", imdbUrl: "https://www.imdb.com/title/tt5205548/" },
    { title: "Laali Haadu", language: "Kannada", imdbUrl: "https://www.imdb.com/title/tt0190574/" },
    { title: "Lodge", language: "Kannada", imdbUrl: "https://www.imdb.com/title/tt12535264/" },
    { title: "Mummy", language: "Kannada", imdbUrl: "https://www.imdb.com/title/tt6272364/" },
    { title: "No. 66 Madura Bus", language: "Kannada", imdbUrl: "https://www.imdb.com/title/tt2578244/" },
    { title: "Prayoga", language: "Kannada", imdbUrl: "https://www.imdb.com/title/tt0861689/" },
    { title: "Shivasharane Siri", language: "Kannada", imdbUrl: "https://www.imdb.com/title/tt2685332/" },
    { title: "Swetha", language: "Kannada", imdbUrl: "https://www.imdb.com/title/tt0859663/" }
];


// Tamil Horror Movies
var tamilHorrorMovies = [
    { title: "Rajnayak", language: "Tamil", imdbUrl: "https://www.imdb.com/title/tt7359428/" },
    { title: "Ratsasan", language: "Tamil", imdbUrl: "https://www.imdb.com/title/tt7060344/" },
    { title: "Aval", language: "Tamil", imdbUrl: "https://www.imdb.com/title/tt6152580/" },
    { title: "Demonte Colony", language: "Tamil", imdbUrl: "https://www.imdb.com/title/tt4515078/" },
    { title: "Pizza", language: "Tamil", imdbUrl: "https://www.imdb.com/title/tt2306721/" },
    { title: "Dhilluku Dhuddu", language: "Tamil", imdbUrl: "https://www.imdb.com/title/tt5890492/" },
    { title: "Kanchana", language: "Tamil", imdbUrl: "https://www.imdb.com/title/tt2071620/" },
    { title: "Kanchana 2", language: "Tamil", imdbUrl: "https://www.imdb.com/title/tt3508840/" },
    { title: "Aranmanai", language: "Tamil", imdbUrl: "https://www.imdb.com/title/tt4098724/" },
    { title: "Maya", language: "Tamil", imdbUrl: "https://www.imdb.com/title/tt5080586/" },
    { title: "Yaamirukka Bayamey", language: "Tamil", imdbUrl: "https://www.imdb.com/title/tt3619854/" },
    { title: "Jackson Durai", language: "Tamil", imdbUrl: "https://www.imdb.com/title/tt5785970/" },
    { title: "Devi", language: "Tamil", imdbUrl: "https://www.imdb.com/title/tt6040012/" },
    { title: "Sowkarpettai", language: "Tamil", imdbUrl: "https://www.imdb.com/title/tt5424608/" },
    { title: "Maragadha Naanayam", language: "Tamil", imdbUrl: "https://www.imdb.com/title/tt6221036/" },
    { title: "Kalam", language: "Tamil", imdbUrl: "https://www.imdb.com/title/tt4870828/" },
    { title: "Katteri", language: "Tamil", imdbUrl: "https://www.imdb.com/title/tt12305646/" },
    { title: "Kalpana House", language: "Tamil", imdbUrl: "https://www.imdb.com/title/tt12022018/" },
    { title: "Neeya 2", language: "Tamil", imdbUrl: "https://www.imdb.com/title/tt9875084/" },
    { title: "Dhayam", language: "Tamil", imdbUrl: "https://www.imdb.com/title/tt6094090/" },
    { title: "Sangili Bungili Kadhava Thorae", language: "Tamil", imdbUrl: "https://www.imdb.com/title/tt6240884/" },
    { title: "Unakkenna Venum Sollu", language: "Tamil", imdbUrl: "https://www.imdb.com/title/tt4662900/" },
    { title: "E", language: "Tamil", imdbUrl: "https://www.imdb.com/title/tt5137372/" },
    { title: "Uru", language: "Tamil", imdbUrl: "https://www.imdb.com/title/tt5471378/" },
    { title: "Sadhuram 2", language: "Tamil", imdbUrl: "https://www.imdb.com/title/tt4621028/" },
    { title: "Arundhati", language: "Tamil", imdbUrl: "https://www.imdb.com/title/tt1340867/" },
    { title: "Iruttu", language: "Tamil", imdbUrl: "https://www.imdb.com/title/tt9805068/" },
    { title: "Muni", language: "Tamil", imdbUrl: "https://www.imdb.com/title/tt0758439/" },
    { title: "Dora", language: "Tamil", imdbUrl: "https://www.imdb.com/title/tt6442274/" },
    { title: "Demonte Colony", language: "Tamil", imdbUrl: "https://www.imdb.com/title/tt4515078/" }
];
















from __future__ import division, print_function
#import sys
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras
import cv2
#import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import statistics as st


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index1.html")


























@app.route('/templates/buttons', methods = ['GET','POST'])
def buttons():
    return render_template("buttons.html")

@app.route('/movies/surprise', methods = ['GET', 'POST'])
def moviesSurprise():
    return render_template("moviesSurprise.html")

@app.route('/movies/angry', methods = ['GET', 'POST'])
def moviesAngry():
    return render_template("moviesAngry.html")

@app.route('/movies/sad', methods = ['GET', 'POST'])
def moviesSad():
    return render_template("moviesSad.html")

@app.route('/movies/disgust', methods = ['GET', 'POST'])
def moviesDisgust():
    return render_template("moviesDisgust.html")

@app.route('/movies/happy', methods = ['GET', 'POST'])
def moviesHappy():
    return render_template("moviesHappy.html")

@app.route('/movies/fear', methods = ['GET', 'POST'])
def moviesFear():
    return render_template("moviesFear.html")

@app.route('/movies/neutral', methods = ['GET', 'POST'])
def moviesNeutral():
    return render_template("moviesNeutral.html")
























@app.route('/camera', methods = ['GET', 'POST'])
def camera():
    i=0

    GR_dict={0:(0,255,0),1:(0,0,255)}
    model = tf.keras.models.load_model('final_model.h5')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    output=[]
    cap = cv2.VideoCapture(0)
    while (i<=30):
        ret, img = cap.read()
        faces = face_cascade.detectMultiScale(img,1.05,5)

        for x,y,w,h in faces:

            face_img = img[y:y+h,x:x+w]

            resized = cv2.resize(face_img,(224,224))
            reshaped=resized.reshape(1, 224,224,3)/255
            predictions = model.predict(reshaped)

            max_index = np.argmax(predictions[0])

            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'neutral', 'surprise')
            predicted_emotion = emotions[max_index]
            output.append(predicted_emotion)



            cv2.rectangle(img,(x,y),(x+w,y+h),GR_dict[1],2)
            cv2.rectangle(img,(x,y-40),(x+w,y),GR_dict[1],-1)
            cv2.putText(img, predicted_emotion, (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        i = i+1

        cv2.imshow('LIVE', img)
        key = cv2.waitKey(1)
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            break
    print(output)
    cap.release()
    cv2.destroyAllWindows()
    final_output1 = st.mode(output)
    return render_template("buttons.html",final_output=final_output1)






































emotions_songs = {
    'surprise': ["Surprise Song 1", "Surprise Song 2", "Surprise Song 3"],
    'angry': ["Angry Song 1", "Angry Song 2", "Angry Song 3"],
    'sad': ["Sad Song 1", "Sad Song 2", "Sad Song 3"],
    'disgust': ["Disgust Song 1", "Disgust Song 2", "Disgust Song 3"],
    'happy': ["Happy Song 1", "Happy Song 2", "Happy Song 3"],
    'fear': ["Fear Song 1", "Fear Song 2", "Fear Song 3"],
    'neutral': ["Neutral Song 1", "Neutral Song 2", "Neutral Song 3"]
}

# Route to render the main songs.html page
@app.route('/')
def index():
    return render_template("templates/songsHappy.html", emotions=emotions_songs)







@app.route('/songs/surprise', methods = ['GET', 'POST'])
def songsSurprise():
    return render_template("songsSurprise.html")

@app.route('/songs/angry', methods = ['GET', 'POST'])
def songsAngry():
    return render_template("songs.html")

@app.route('/songs/sad', methods = ['GET', 'POST'])
def songsSad():
    return render_template("songs.html")

@app.route('/songs/disgust', methods = ['GET', 'POST'])
def songsDisgust():
    return render_template("songs.html")

@app.route('/songs/happy', methods = ['GET', 'POST'])
def songsHappy():
    return render_template("songs.html")

@app.route('/songs/fear', methods = ['GET', 'POST'])
def songsFear():
    return render_template("songs.html")

@app.route('/songs/neutral', methods = ['GET', 'POST'])
def songsNeutral():
    return render_template("songs.html")

@app.route('/templates/join_page', methods = ['GET', 'POST'])
def join():
    return render_template("join_page.html")


















from __future__ import division, print_function
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import statistics as st


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index1.html")
    
    
@app.route('/camera', methods=['GET', 'POST'])
def camera():
    i = 0

    GR_dict = {0: (0, 255, 0), 1: (0, 0, 255)}
    model = tf.keras.models.load_model('emotion_model.h5')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    output = []
    cap = cv2.VideoCapture(0)
    while (i <= 30):
        ret, img = cap.read()
        faces = face_cascade.detectMultiScale(img, 1.05, 5)

        for x, y, w, h in faces:
            face_img = img[y:y + h, x:x + w]
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            resized = cv2.resize(gray_face, (48, 48))  # Resize to (48, 48)
            reshaped = resized.reshape(1, 48, 48, 1) / 255.0  # Reshape and normalize
            predictions = model.predict(reshaped)

            max_index = np.argmax(predictions[0])

            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'neutral', 'surprise')
            predicted_emotion = emotions[max_index]
            output.append(predicted_emotion)

            cv2.rectangle(img, (x, y), (x + w, y + h), GR_dict[1], 2)
            cv2.rectangle(img, (x, y - 40), (x + w, y), GR_dict[1], -1)
            cv2.putText(img, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        i = i + 1

        cv2.imshow('LIVE', img)
        key = cv2.waitKey(1)
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            break
    print(output)
    cap.release()
    cv2.destroyAllWindows()
    final_output1 = st.mode(output)
    return render_template("buttons.html", final_output=final_output1)


@app.route('/templates/buttons', methods=['GET', 'POST'])
def buttons():
    return render_template("buttons.html")

@app.route('/movies/surprise', methods=['GET', 'POST'])
def moviesSurprise():
    return render_template("moviesSurprise.html")

@app.route('/movies/angry', methods=['GET', 'POST'])
def moviesAngry():
    return render_template("moviesAngry.html")

@app.route('/movies/sad', methods=['GET', 'POST'])
def moviesSad():
    return render_template("moviesSad.html")

@app.route('/movies/disgust', methods=['GET', 'POST'])
def moviesDisgust():
    return render_template("moviesDisgust.html")

@app.route('/movies/happy', methods=['GET', 'POST'])
def moviesHappy():
    return render_template("moviesHappy.html")

@app.route('/movies/fear', methods=['GET', 'POST'])
def moviesFear():
    return render_template("moviesFear.html")

@app.route('/movies/neutral', methods=['GET', 'POST'])
def moviesNeutral():
    return render_template("moviesNeutral.html")

@app.route('/songs/surprise', methods=['GET', 'POST'])
def songsSurprise():
    return render_template("songsSurprise.html")

@app.route('/songs/angry', methods=['GET', 'POST'])
def songsAngry():
    return render_template("songsAngry.html")

@app.route('/songs/sad', methods=['GET', 'POST'])
def songsSad():
    return render_template("songsSad.html")

@app.route('/songs/disgust', methods=['GET', 'POST'])
def songsDisgust():
    return render_template("songsDisgust.html")

@app.route('/songs/happy', methods=['GET', 'POST'])
def songsHappy():
    return render_template("songsHappy.html")

@app.route('/songs/fear', methods=['GET', 'POST'])
def songsFear():
    return render_template("songsFear.html")

@app.route('/songs/neutral', methods=['GET', 'POST'])
def songsNeutral():
    return render_template("songsSad.html")

@app.route('/templates/join_page', methods=['GET', 'POST'])
def join():
    return render_template("join_page.html")

if __name__ == "__main__":
    app.run(debug=True)
