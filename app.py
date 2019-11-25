from flask import Flask, request, jsonify
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from PIL import Image
import datetime
import tensorflow as tf
from gender_age_classifier import DetectGenderAge

app = Flask(__name__)

global graph

graph = tf.get_default_graph()

@app.route('/api', methods=['POST'])
def face_emotion():
    if request.method == "POST":
        if request.files:
            
            imagem  = request.files['media']
            
            npimg = np.fromstring(imagem.read(), np.uint8)
            
            # converter o numpy array para imagem
            img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
            img_colorida = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            
            with graph.as_default():
                
                try:
                    genero, genero_est, idade, idade_est = DetectGenderAge(img_colorida)
                    emocoes = DetectEmotions(img)
                
                    nome_emocao = []
                    emocoes_resultados = []
                    probabilidade_emocao = []
                    for chave, valor in emocoes.items():
                        nome_emocao.append(chave)
                        emocoes_resultados.append(True if valor >= 20.0 else False)
                        probabilidade_emocao.append(valor)
                    
                    dicionario_emocoes = {
                    nome_emocao[0] : {
                        'valor': emocoes_resultados[0],
                        'confiança': probabilidade_emocao[0]
                    },
                    nome_emocao[1] : {
                        'valor': emocoes_resultados[1],
                        'confiança': probabilidade_emocao[1]
                    },
                    nome_emocao[2] : {
                        'valor': emocoes_resultados[2],
                        'confiança': probabilidade_emocao[2]
                    },
                    nome_emocao[3] : {
                        'valor': emocoes_resultados[3],
                        'confiança': probabilidade_emocao[3]
                    },
                    nome_emocao[4] : {
                        'valor': emocoes_resultados[4],
                        'confiança': probabilidade_emocao[4]
                    },
                    nome_emocao[5] : {
                        'valor': emocoes_resultados[5],
                        'confiança': probabilidade_emocao[5]
                    },
                    nome_emocao[6] : {
                        'valor': emocoes_resultados[6],
                        'confiança': probabilidade_emocao[6]
                    }
                }

                    retorno_texto = {
                        'status': 'sucesso',
                        'foto': {
                            'nome': imagem.filename,
                            'largura': img_colorida.shape[1],
                            'altura': img_colorida.shape[0]
                            },
                        'sentimentos' : dicionario_emocoes,
                        'gênero' : {
                            'valor': genero, 
                            'confiança': genero_est
                            },
                        'idade': {
                            'valor': idade, 
                            'confiança': idade_est
                            },
                        'data' : str(datetime.datetime.now())
                        }
                except:
                    retorno_texto = {
                        'status': 'erro'
                    }
    return jsonify(resposta=retorno_texto)

def DetectEmotions(imagem):
    
    #modelo que reconhece o rosto em foto
    classificador_facial = cv2.CascadeClassifier('models/frontal/haarcascade_frontalface_default.xml')
    
    #inicializa o modelo com os pesos
    classificador = model_from_json(open('models/emotion/face_emotion_classifier.json', 'r').read())
    classificador.load_weights('models/emotion/face_emotion_classifier.h5')
    
    tamanho = (imagem.shape[1], imagem.shape[0])
    
    frame = cv2.resize(imagem, tamanho)
    
    faces_detectadas = classificador_facial.detectMultiScale(frame)
    
    for (x,y,width,height) in faces_detectadas:
        rosto_detectado = imagem[y:y+width, x:x+height]
        rosto_detectado = cv2.resize(rosto_detectado,(48,48))
        pixels = img_to_array(Image.fromarray(rosto_detectado))
        pixels = np.expand_dims(pixels, axis=0)
        pixels /= 255
        
        # predição
        # 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
        predicao = classificador.predict(pixels)[0]
        predicao = list(np.around(np.array(predicao*100),2))
        
        predicao_saida = [round(float(item),2) for item in predicao]
        
        rotulos = ["irritação", "náusea", "medo", "felicidade", "tristeza", "surpresa", "neutralidade"]
        
        resultado = dict(zip(rotulos, predicao_saida))
        
    return resultado

if __name__ == '__main__':
    app.config["JSON_SORT_KEYS"] = False
    app.run(port=9000, debug=True)
