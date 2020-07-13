import face_recognition
import numpy as np
from PIL import Image, ImageDraw
from IPython.display import display
import os
import cv2

def Union(lst1, lst2): 
    final_list = list(set(lst1) | set(lst2)) 
    return final_list

def Diff(li1, li2): 
    return (list(set(li1) - set(li2)))

pathOut = r"C:/Users/RoG/Desktop/github/test/"
count = 0
counter = 1
listing = os.listdir(r'C:/Users/RoG/Desktop/github/testvid')
for vid in listing:
    vid = r"C:/Users/RoG/Desktop/github/testvid/"+vid
    cap = cv2.VideoCapture(vid)
    count = 0
    counter += 1
    success = True
    while success:
        success,image = cap.read()
        if count%152 == 0 :
            print('read a new frame:',success)
            c=count/152 #(interval in sec)*(fps)
            cv2.imwrite(pathOut + 'image%d.jpg'%c,image)
        count+=1
# The program we will be finding faces on the example below
#pil_im = Image.open('image.jpg')
#pil_im.show()


# Load a sample picture and learn how to recognize it.
person1 = face_recognition.load_image_file("train/ronaldo.jpg")
person1_face_encoding = face_recognition.face_encodings(person1)[0]

person2 = face_recognition.load_image_file("train/messi.jpg")
person2_face_encoding = face_recognition.face_encodings(person2)[0]

person3 = face_recognition.load_image_file("train/rooney.jpg")
person3_face_encoding = face_recognition.face_encodings(person3)[0]

person4 = face_recognition.load_image_file("train/mbappe.jpg")
person4_face_encoding = face_recognition.face_encodings(person4)[0]

person5 = face_recognition.load_image_file("train/neymar.jpg")
person5_face_encoding = face_recognition.face_encodings(person5)[0]

person6 = face_recognition.load_image_file("train/drogba.png")
person6_face_encoding = face_recognition.face_encodings(person6)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    person1_face_encoding,
    person2_face_encoding,
    person3_face_encoding,
    person4_face_encoding,
    person5_face_encoding,
    person6_face_encoding
]
known_face_names = [
    "Ronaldo",
    "Messi",
    "Rooney",
    "Mbappe",
    "Neymar",
    "Drogba"
]
print('Trained for', len(known_face_encodings), 'faces.')







rec_list=[]

listing = os.listdir(r'C:/Users/RoG/Desktop/github/test')
lnum=len(listing)
for i in range(lnum):

    list1=[]
    
    # Load an image with an unknown face
    unknown_image = face_recognition.load_image_file("test/image%d.jpg"%i)

    # Find all the faces and face encodings in the unknown image
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

    # Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
    # See http://pillow.readthedocs.io/ for more about PIL/Pillow
    pil_image = Image.fromarray(unknown_image)
    # Create a Pillow ImageDraw Draw instance to draw with
    draw = ImageDraw.Draw(pil_image)

    

    # Loop through each face found in the unknown image
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            list1.append(str(known_face_names[best_match_index]))
            name = known_face_names[best_match_index]

        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

        rec_list=Union(rec_list,list1)

    # Display the resulting image
    pil_image.show()

        
alist=[]
alist=Diff(known_face_names,rec_list)
# Remove the drawing library from memory as per the Pillow docs
del draw
#print(rec_list)
f1 = open("List-Present.txt","w+")
for i in range(len(rec_list)):
    f1.write(str(rec_list[i]) + "\n")
f2 = open("List-Absent.txt","w+")
for i in range(len(alist)):
    f2.write(str(alist[i]) + "\n")


