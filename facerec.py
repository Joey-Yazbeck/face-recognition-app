import face_recognition
import cv2
import psycopg2
import config
import numpy
import automated_email2
from datetime import date

#This will return video from the first webcam on your computer
video_capture = cv2.VideoCapture(0)

#connect to the db
con = psycopg2.connect(
            database=config.database,
            user = config.username,
            password = config.password)
cur = con.cursor()

cur.execute('SELECT ph."Photo", pr."FullName" FROM photo ph JOIN target t ON ph."TargetId" = t."TargetId" JOIN profile pr ON t."ProfileId" = pr."ProfileId";')
# cur.ex ecute('select "Photo","TargetId" from photo')
rows = cur.fetchall()


known_face_names=[]
known_face_encodings =[]

for r in rows:
    known_face_encodings.append(face_recognition.face_encodings(face_recognition.load_image_file("Images/" + r[0]))[0])

    known_face_names.append(r[1])

#----------------------------------------------------------------------------------------------------------------------

while True:
    ret, frame = video_capture.read()

    rgb_frame = numpy.ascontiguousarray(frame[:, :, ::-1])

    face_locations = face_recognition.face_locations(rgb_frame,model='hog')
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding,0.4)

        name = "Random Person"
       
        if True in matches:
            current_date = date.today()
            first_match_index = matches.index(True)
            
            name = known_face_names[first_match_index]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255),2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

           

            subquery = '''SELECT t."TargetId" from target t JOIN profile p on t."ProfileId" = p."ProfileId" where p."FullName" = %s'''
	
            query = '''INSERT INTO public.alerts("AlertId", "IsRead", "TargetId", "Message", "DateCreated") VALUES (DEFAULT,False, (%s), %s,%s) '''
            cur.execute(subquery, (name,))
            subquery_result = cur.fetchone()
            ValuetoCheck = subquery_result
            select_query = '''SELECT "IsRead", "TargetId" FROM public.alerts WHERE "IsRead" = False AND "TargetId" = %s'''

            # Execute the query with the value as a parameter
            cur.execute(select_query, (ValuetoCheck,))
            result = cur.fetchone()
            if result is None:
                insert_data = (subquery_result, 'Target Detected',current_date,)
                cur.execute(query, insert_data)

                con.commit()
                automated_email2.send_email2(name)
                if (name != "Random Person"):
                    print(name, "was here")


        # Close the cursor and database connection
        #cur.close()
        #con.close()

        # automated_email.send_email(name)
        
        
            
    cv2.imshow('Video', frame)

    # Q to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
