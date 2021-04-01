import cv2
import numpy as np
from PIL import Image
import pytesseract


def sayacdondur(path):

    image = cv2.imread(path) #resmi okuduk

    r=800 / image.shape[1]
    dim=(800,int(image.shape[0] * r))

    img = cv2.resize(image,dim,interpolation=cv2.INTER_AREA)

    # resim en ve boylarına ulaşacağız

    #edged=cv2.Canny(img,30,200)



    img_width=img.shape[1] #genişlik ikinci
    img_height=img.shape[0] #boy ilk dizi


    # görüntüyü modele vereceğiz
    # görüntüyü blob formata yani 4 boyutlu tensöre çevireceğiz
    #crop kırpmak

    img_blob=cv2.dnn.blobFromImage(img,1/255,(416,416),swapRB=True , crop=False)

    labels=["sayac"]

    colors=["0,0,255"]

    colors=[np.array(color.split(",")).astype("int") for color in colors]

    colors=np.array(colors)

    # tile matrisi büyüttük alta doğru 18 defa yana doğru 1 defa
    #colors = np.tile(colors,(18,1))

    # modeli yüklüyoruz

    model=cv2.dnn.readNetFromDarknet("C:/Users/seyit/Desktop/YOLO/sayac/spot_yolov4.cfg","C:/Users/seyit/Desktop/YOLO/sayac/spot_yolov4_1000.weights")

    layers=model.getLayerNames()

    output_layer=[layers[layer[0]-1] for layer in model.getUnconnectedOutLayers()]


    model.setInput(img_blob)

    detection_layers=model.forward(output_layer)

    ids_list = []
    boxes_list = []
    confidences_list = []


    for detection_layer in detection_layers:
        for object_detection in detection_layer:

            scores = object_detection[5:]
            predicted_id = np.argmax(scores)
            confidence = scores[predicted_id]

            if confidence > 0.20:

                label = labels[predicted_id]
                bounding_box = object_detection[0:4] * np.array([img_width,img_height,img_width,img_height])
                (box_center_x, box_center_y, box_width, box_height) = bounding_box.astype("int")

                start_x = int(box_center_x - (box_width/2))
                start_y = int(box_center_y - (box_height/2))


                ############## NON-MAXIMUM SUPPRESSION - OPERATION 2 ###################

                ids_list.append(predicted_id)
                confidences_list.append(float(confidence))
                boxes_list.append([start_x, start_y, int(box_width), int(box_height)])

                ############################ END OF OPERATION 2 ########################



    ############## NON-MAXIMUM SUPPRESSION - OPERATION 3 ###################

    max_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)

    for max_id in max_ids:

        max_class_id = max_id[0]
        box = boxes_list[max_class_id]

        start_x = box[0]
        start_y = box[1]
        box_width = box[2]
        box_height = box[3]

        predicted_id = ids_list[max_class_id]
        label = labels[predicted_id]
        confidence = confidences_list[max_class_id]

        ############################ END OF OPERATION 3 ########################

        end_x = start_x + box_width
        end_y = start_y + box_height

        box_color = colors[predicted_id]
        box_color = [int(each) for each in box_color]


        label = "{}: {:.2f}%".format(label, confidence*100)
        print("predicted object {}".format(label))


        cv2.rectangle(img, (start_x,start_y),(end_x,end_y),box_color,1)
        cv2.putText(img,label,(start_x,start_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)



    print(box)



    resim=img[start_y:start_y+ box_height , start_x:start_x+box_width]
    edged=cv2.Canny(resim,250,50)
    cv2.imwrite('../kayit.png', edged)

    # okuma işlemi
    #img_to_str = pytesseract.pytesseract.tesseract_cmd=r"sayac/Tesseract-OCR/tesseract.exe"
    imgg=Image.open("../kayit.png")
    result=pytesseract.image_to_string(imgg)

    with open("../okuma.txt", mode="w") as file:
        file.write(result)
        print("ready !!")
    return result

#cv2.imshow("Detection Window", img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
