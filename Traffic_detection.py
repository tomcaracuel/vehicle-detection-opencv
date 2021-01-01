# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 16:04:11 2020

@author: Tomás


Se procesa video (o cámara en tiempo real) del tráfico. Se hace un conteo de la cantidad
de vehíchulos que se dirigen al norte, al sur y en total.
Además se genera un archivo csv que guarda dichos datos (cantidad de vehiculos, hora, dirección)
"""

import cv2
import numpy as np
from time import sleep
import datetime
import csv

length_min=80 #largo mínimo rectangulo
height_min=80 #altura mínima retangulo

offset=6 #error pixeles 

pos_lin = 550 #posicion de la linea (eje y) 
al_sur = (25, 500)  #linea carril sur
al_norte = (700, 1200) #linea carril norte

delay = 60 #FPS del video

detec = []
vehiculos = 0
detect_sur = []
detec_norte = []
vehiculos_sur = 0
vehiculos_norte = 0

#rectangulo identificador de vehiculos
def centro_rect(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    
    return cx,cy

#captura de video
cap = cv2.VideoCapture('video.mp4')
sust = cv2.bgsegm.createBackgroundSubtractorMOG()


#funcion para generar csv 
def write_csv(data):
    with open('trafico_ruta.csv', 'a') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(data)


#Procesamiento de la imagen
while True:
    ret , frame1 = cap.read()
    tempo = float(1/delay)
    sleep(tempo) 
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3),5)
    img_sub = sust.apply(blur)
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx (dilat, cv2. MORPH_CLOSE , kernel)
    dilatada = cv2.morphologyEx (dilatada, cv2. MORPH_CLOSE , kernel)
    contorno,h=cv2.findContours(dilatada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.line(frame1, (al_sur[0], pos_lin), (al_sur[1], pos_lin), (255,127,0), 3) #linea carril sur
    cv2.line(frame1, (al_norte[0], pos_lin), (al_norte[1], pos_lin), (0,50,255), 3) #linea carril norte
    for(i,c) in enumerate(contorno):
        (x,y,w,h) = cv2.boundingRect(c)
        validar_contorno = (w >= length_min) and (h >= height_min)
        if not validar_contorno:
            continue    
        
        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)        
        centro = centro_rect(x, y, w, h)
        detec.append(centro)
        cv2.circle(frame1, centro, 4, (0, 0,255), -1)
        
        
        for (x,y) in detec: #contador de autos QUE VAN AL SUR
            if y<(pos_lin + offset) and y>(pos_lin - offset) and x>(al_sur[0] + offset) and x<(al_sur[1] + offset):
                vehiculos_sur += 1
                cv2.line(frame1, (al_sur[0], pos_lin), (al_sur[1], pos_lin), (0,127,255), 3)  
                detec.remove((x,y))
                print("car to south is detected : "+str(vehiculos_sur))  
                write_csv([+(vehiculos_sur), datetime.datetime.now(), 'SUR']) #archivo csv [cant autos, hora, direccion]
                        

        for (x,y) in detec: #contador de autos QUE VAN AL NORTE
            if y<(pos_lin + offset) and y>(pos_lin - offset) and x>(al_norte[0] + offset) and x<(al_norte[1] + offset):
                vehiculos_norte += 1
                cv2.line(frame1, (al_norte[0], pos_lin), (al_norte[1], pos_lin), (0,127,255), 3)  
                detec.remove((x,y))
                print("car to north is detected : "+str(vehiculos_norte))   
                write_csv([+(vehiculos_norte), datetime.datetime.now(), 'NORTE']) #archivo csv [cant autos, hora, direccion]
                
  
    
    # Obtencion fecha y hora     
    dt = str(datetime.datetime.now()) 
     
    
    #Mostrar fecha y hora  
    font = cv2.FONT_HERSHEY_DUPLEX 
    cv2.putText(frame1, dt, 
                            (10, 700), 
                            font, 0.5, 
                            (0, 255, 0),  
                            1, cv2.LINE_8)
        
    #Mostrar contador de autos 
    cv2.putText(frame1, "VEHICLE TO NORT COUNT : "+str(vehiculos_norte), (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,50,255),2)
    cv2.putText(frame1, "VEHICLE TO SOUTH COUNT : "+str(vehiculos_sur), (150, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,127,0),2)
    cv2.putText(frame1, "VEHICLE TOTAL COUNT : "+str(vehiculos_sur + vehiculos_norte), (750, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 225, 0),2)
    
    cv2.imshow("Video Original" , frame1)
    cv2.imshow("Detectar",dilatada)
    
    
 

    if cv2.waitKey(1) == 27:
        break
    

    
    
cv2.destroyAllWindows()
cap.release()