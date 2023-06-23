import speech_recognition as sr
from datetime import date
import psycopg2
import config
import capture
import automated_email

# Set up the database connection
db = psycopg2.connect(database=config.database,
            user = config.username,
            password = config.password)
cursor = db.cursor()

# Create a speech recognizer object
recognizer = sr.Recognizer()

# Use the default microphone as the audio source
with sr.Microphone() as source:
    print("Speak something...")
    audio = recognizer.listen(source)

    try:
        # Perform speech recognition
        text = recognizer.recognize_google(audio)
        print("Speech recognized:", text)

        # Insert speech text into the database
        
        current_datetime = date.today()

        select_query = 'SELECT * FROM keyword where keyword."Keyword" = %s'
        cursor.execute(select_query,(text,))
        # db.commit()
        result = cursor.fetchone()
      

        if result:
            capture.capture_image()
            automated_email.send_email()
            cursor2 = db.cursor()
            # subquery = '''select max(photo."PhotoId") from photo where photo."Photo" like 'Captured_image%' '''
            # insert_query2 = '''INSERT INTO public.suspect ("SuspectId", "CaptureDate", "CameraId", "PhotoId") VALUES (DEFAULT,%s,1,({0}));'''.format(subquery)
            # cursor2.execute(insert_query2,(timestamp,))
            # db.commit()

            subquery = '''SELECT max(photo."PhotoId") FROM photo WHERE photo."Photo" LIKE 'Captured_image%' '''
            insert_query2 = '''INSERT INTO public.suspect ("SuspectId", "CaptureDate", "CameraId", "PhotoId") Values (DEFAULT, %s, 1, (%s))'''
            cursor2.execute(subquery,)
            subquery_result = cursor2.fetchone()
            insert_data = (current_datetime,subquery_result,)
            cursor2.execute(insert_query2, insert_data )
            query = '''INSERT INTO public.alerts("AlertId", "IsRead", "Message", "DateCreated") VALUES (DEFAULT,False,'Suspect Detected',%s) '''
            cursor2.execute(query, (current_datetime,))
            db.commit()

            
            
        # insert_query = "INSERT INTO message (speech_text, timestamp) VALUES (%s, %s)"
        # cursor.execute(insert_query, (text, timestamp))
        # db.commit()
        # print("Speech text inserted into the database.")
        # capture.capture_image()

    except sr.UnknownValueError:
        print("Speech recognition could not understand audio.")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

# Close the database connection
cursor.close()
db.close()
