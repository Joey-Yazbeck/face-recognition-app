import psycopg2
import smtplib
import config


con = psycopg2.connect(
    database=config.database,
    user = config.username,
    password = config.password)
cur = con.cursor()
    # Execute your specific query to fetch the receiver emails from the database
cur.execute('select "Email" from "user";')
receiver_emails = [row[0] for row in cur.fetchall()]
    
cur.close()
con.close()
    
  

def send_email():
   
        sender_email = "smartmonitoringsystemm@gmail.com"  # Replace with your email address
        password = "ktjmhjwnvglowwya"  # Replace with your email password
    
        for receiver_email in receiver_emails:
            # Construct the email headers
            headers = f"From: {sender_email}\r\nTo: {receiver_email}\r\nSubject: New Alert\r\n"
            
            # Construct the email body
            
            email_body = f"{headers}\r\n Suspect is Detected!"
            try:
                # Establish a secure connection with the SMTP server
                server = smtplib.SMTP("smtp.gmail.com", 587)
                server.starttls()
                server.login(sender_email, password)
                
                # Send the email
                server.sendmail(sender_email, receiver_email, email_body)
                print(f"Email sent successfully to {receiver_email}!")
            except Exception as e:
                print(f"An error occurred while sending the email to {receiver_email}: {str(e)}")
            finally:
                # Close the connection
                server.quit()

   
    
    
       