# 필요한 라이브러리 임포트
import paho.mqtt.client as mqtt
import mysql.connector
import json

# ★★★★★★★★★★★★★★★★★★★★★ 설정 부분 ★★★★★★★★★★★★★★★★★★★★★

# --- MQTT 브로커 정보 (라즈베리파이) ---
MQTT_BROKER_IP = "100.112.74.119"
MQTT_PORT = 1883
MQTT_TOPIC = "AjouHospital/#"
MQTT_USER = "mqttuser"
MQTT_PASSWORD = "asdf"

# --- MySQL 데이터베이스 정보 (노트북) ---
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = "kwangyeon404@"
DB_NAME = "mqtt_data"

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

def on_connect(client, userdata, flags, rc):
    """브로커 연결 성공 시 실행되는 함수"""
    if rc == 0:
        print(f"✅ 라즈베리파이 브로커({MQTT_BROKER_IP})에 연결 성공!")
        client.subscribe(MQTT_TOPIC)
        print(f"'{MQTT_TOPIC}' 토픽 구독을 시작합니다.")
    else:
        print(f"❌ 브로커 연결 실패! (코드: {rc}) - IP주소나 포트, 계정 정보를 확인하세요.")

def on_message(client, userdata, msg):
    """메시지 수신 시 DB에 저장하는 함수"""
    topic = msg.topic
    payload = msg.payload.decode('utf-8')
    print(f"📨 메시지 수신 [토픽: {topic}] > {payload}")

    db_conn = None  # finally 블록에서 사용하기 위해 미리 선언
    cursor = None   # finally 블록에서 사용하기 위해 미리 선언
    try:
        # 토픽 파싱
        parts = topic.split('/')
        hospital = parts[0] if len(parts) > 0 else ""
        room_no = parts[1] if len(parts) > 1 else ""
        bed_no = parts[2] if len(parts) > 2 else ""

        # JSON 페이로드 파싱
        data = json.loads(payload)
        timestamp = data.get("timestamp", None)
        spo2 = data.get("spo2", None)
        heartrate = data.get("heartrate", None)

        # 데이터베이스 연결
        db_conn = mysql.connector.connect(
            host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME
        )
        cursor = db_conn.cursor()

        # SQL 쿼리 실행
        sql = """
            INSERT INTO SmartRingData (hospital, room_no, bed_no, timestamp, spo2, heartrate)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        values = (hospital, room_no, bed_no, timestamp, spo2, heartrate)
        cursor.execute(sql, values)
        db_conn.commit()

        print("💾 MySQL에 저장 완료!")

    except json.JSONDecodeError:
        print(f"❌ JSON 파싱 실패: {payload}")
    except mysql.connector.Error as err:
        print(f"❌ 데이터베이스 오류: {err}")
    except Exception as e:
        print(f"❌ 처리 중 알 수 없는 오류 발생: {e}")
    finally:
        # 연결이 성공했을 경우에만 커서와 연결을 닫음
        if cursor:
            cursor.close()
        if db_conn and db_conn.is_connected():
            db_conn.close()

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