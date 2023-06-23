import cv2
import datetime
import config
import psycopg2

def capture_image():
    
    
    # Open the default camera (index 0)
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Unable to open the camera.")
        return

    # Read and display the captured frame
    ret, frame = cap.read()
    if ret:
        

         # Generate the file name with the current date and time
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        image_filename = "Captured_image_" + formatted_datetime + ".jpg"

        cv2.imshow(image_filename, frame)
        cv2.waitKey(0)

        # Save the captured image to disk
        # image_path = "captured_image.jpg"
        image_path = r"Images/"+image_filename
        cv2.imwrite(image_path, frame)
        print("Image saved successfully:", image_path)
        
        db = psycopg2.connect(database=config.database,
            user = config.username,
            password = config.password)
        cursor = db.cursor()
        insert_query = '''INSERT INTO public.photo ("PhotoId", "Photo") VALUES (DEFAULT,%s);'''
        cursor.execute(insert_query,(image_filename,))
        db.commit()

        # cursor.execute('INSERT INTO photo values  ;')

    # Release the camera
    cap.release()
    cv2.destroyAllWindows()

