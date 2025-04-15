import os
import smtplib  # SMTP 사용을 위한 모듈
import re  # Regular Expression을 활용하기 위한 모듈
import glob
from email.mime.multipart import MIMEMultipart  # 메일의 Data 영역의 메시지를 만드는 모듈
from email.mime.text import MIMEText  # 메일의 본문 내용을 만드는 모듈
from email.mime.image import MIMEImage  # 메일의 이미지 파일을 base64 형식으로 변환하기 위한 모듈
from email.mime.application import MIMEApplication  # CSV 파일 첨부를 위한 모듈 추가
from dotenv import load_dotenv



class GmailSender: 
    def __init__(self):
        # load APP password
        load_dotenv('./')
        self.my_account = os.environ.get('MY_GMAIL_ADRESSS')
        self.password = os.environ.get('GOOGLE_APP_PASSWORD')

        # STMP 서버 설정
        self.gmail_smtp = "smtp.gmail.com"
        self.gmail_port = 465

    def connect_smtp(self):
        """SMTP 서버 연결 및 로그인"""
        self.smtp = smtplib.SMTP_SSL(self.gmail_smtp, self.gmail_port)
        self.smtp.login(self.my_account, self.password)

    def create_message(self, subject="오늘의 정치 뉴스"):
        """이메일 기본 설정"""
        self.msg = MIMEMultipart()
        self.msg["Subject"] = subject
        self.msg["From"] = self.my_account
        self.msg["To"] = self.my_account
        
        # 기본 본문 내용
        content = "오늘의 토픽별 정치 뉴스.\n\n데이터를 전달드립니다.\n\n감사합니다\n\n"
        content_part = MIMEText(content, "plain")
        self.msg.attach(content_part)


    def attach_files(self, data_path):
        """차트 이미지와 CSV 파일 첨부"""
        # 차트 이미지 첨부
        chart_path = f"{data_path}/Chart"
        for image_file in glob.glob(f"{chart_path}/*.png"):
            with open(image_file, 'rb') as file:
                image_name = os.path.basename(image_file)
                img = MIMEImage(file.read())
                img.add_header('Content-Disposition', 'attachment', filename=image_name)
                self.msg.attach(img)
                print(f"이미지 첨부 완료: {image_name}")
        
        # CSV 파일 첨부
        documents_path = f"{data_path}/Documents"
        for csv_file in glob.glob(f"{documents_path}/*.csv"):
            with open(csv_file, 'rb') as file:
                csv_name = os.path.basename(csv_file)
                csv_attachment = MIMEApplication(file.read(), _subtype='csv')
                csv_attachment.add_header('Content-Disposition', 'attachment', filename=csv_name)
                self.msg.attach(csv_attachment)
                print(f"CSV 파일 첨부 완료: {csv_name}")

    def close(self):
        """SMTP 서버 연결 해제"""
        self.smtp.quit()


    def send_email(self, to_mail=None):
        """이메일 전송"""
        if to_mail is None:
            to_mail = self.my_account
            
        reg = "^[a-zA-Z0-9.+_-]+@[a-zA-Z0-9]+\.[a-zA-Z]{2,3}$"
        if re.match(reg, to_mail):
            self.smtp.sendmail(self.my_account, to_mail, self.msg.as_string())
            print("정상적으로 메일이 발송되었습니다.")
        else:
            print("받으실 메일 주소를 정확히 입력하십시오.")







def send_topic_results(data_path):
    """토픽 모델링 결과를 이메일로 전송"""
    try:
        # Gmail sender 초기화
        sender = GmailSender()
        
        # SMTP 연결
        sender.connect_smtp()
        
        # 메시지 생성
        sender.create_message()
        
        # 파일 첨부
        sender.attach_files(data_path)
        
        # 이메일 전송
        sender.send_email()
        
    except Exception as e:
        print(f"이메일 전송 중 오류 발생: {str(e)}")
    
    finally:
        # SMTP 연결 해제
        sender.close()