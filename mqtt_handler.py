
# 필요한 라이브러리 임포트
import paho.mqtt.client as mqtt
import mysql.connector
import json
from datetime import datetime # ▼▼▼ [수정] 이 줄을 추가하세요. ▼▼▼

# ★★★★★★★★★★★★★★★★★★★★★ 설정 부분 ★★★★★★★★★★★★★★★★★★★★★

# --- MQTT 브로커 정보 (라즈베리파이) ---
MQTT_BROKER_IP = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "AjouHospital/patient/#"
MQTT_USER = "mqttuser"
MQTT_PASSWORD = "asdf"

# --- MySQL 데이터베이스 정보 (노트북) ---
DB_HOST = "100.68.16.79"
DB_USER = "mqtt_user"
DB_PASSWORD = "Kwangyeon404@"
DB_NAME = "AjouHospital_DB"

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

def on_connect(client, userdata, flags, rc):
    """브로커 연결 성공 시 실행되는 함수"""
    if rc == 0:
        print(f"✅ MQTT 브로커({MQTT_BROKER_IP})에 연결 성공!")
        client.subscribe(MQTT_TOPIC)
        print(f"'{MQTT_TOPIC}' 토픽 구독을 시작합니다.")
    else:
        print(f"❌ 브로커 연결 실패! (코드: {rc})")

def on_message(client, userdata, msg):
    """메시지 수신 시 DB에 저장하는 함수"""
    topic = msg.topic
    payload = msg.payload.decode('utf-8')
    print(f"📨 메시지 수신 [토픽: {topic}] > {payload}")

    db_conn = None
    cursor = None
    try:
        # --- [수정] 토픽에서 patient_id 추출 ---
        parts = topic.split('/')
        if len(parts) < 3 or parts[1] != 'patient':
            print(f"⚠️ 토픽 형식이 올바르지 않습니다: {topic}")
            return
        
        patient_id = int(parts[2]) # 토픽의 마지막 부분을 patient_id로 사용

        # JSON 페이로드 파싱
        data = json.loads(payload)
        # timestamp가 없으면 현재 시간 사용
        timestamp_str = data.get("timestamp", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        spo2 = data.get("spo2")
        heartrate = data.get("heartrate")

        # DB 연결
        db_conn = mysql.connector.connect(
            host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME
        )
        cursor = db_conn.cursor()

        # --- [수정] smartring_logs 테이블에 INSERT ---
        sql = """
            INSERT INTO smartring_logs (patient_id, timestamp, spo2, heartrate)
            VALUES (%s, %s, %s, %s)
        """
        values = (patient_id, timestamp_str, spo2, heartrate)
        cursor.execute(sql, values)
        db_conn.commit()

        print(f"💾 MySQL smartring_logs 테이블에 저장 완료! (환자 ID: {patient_id})")

    except json.JSONDecodeError:
        print(f"❌ JSON 파싱 실패: {payload}")
    except (mysql.connector.Error, ValueError) as err:
        # DB 오류 또는 patient_id 변환 오류
        print(f"❌ 데이터베이스 또는 데이터 처리 오류: {err}")
    except Exception as e:
        print(f"❌ 알 수 없는 오류 발생: {e}")
    finally:
        if cursor: cursor.close()
        if db_conn and db_conn.is_connected(): db_conn.close()

def run_mqtt_client():
    """main.py에서 호출할 MQTT 클라이언트 실행 함수"""
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    client.on_connect = on_connect
    client.on_message = on_message
    client.username_pw_set(MQTT_USER, MQTT_PASSWORD)

    try:
        print("MQTT 클라이언트를 시작합니다...")
        client.connect(MQTT_BROKER_IP, MQTT_PORT, 60)
        client.loop_forever()
    except Exception as e:
        print(f"❌ MQTT 클라이언트 실행 중 오류 발생: {e}")


if __name__ == "__main__":
    run_mqtt_client()
