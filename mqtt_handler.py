# 필요한 라이브러리 임포트
import paho.mqtt.client as mqtt
import mysql.connector
import json
from datetime import datetime

# ★★★★★★★★★★★★★★★★★★★★★ 설정 부분 ★★★★★★★★★★★★★★★★★★★★★

# --- MQTT 브로커 정보 ---
MQTT_BROKER_IP = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "AjouHospital/patient/#"
MQTT_USER = "mqttuser"
MQTT_PASSWORD = "asdf"

# --- MySQL 데이터베이스 정보 ---
DB_HOST = "localhost"
DB_USER = "dashboard_user"
DB_PASSWORD = "Kwangyeon404@"
DB_NAME = "AjouHospital_DB"

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

# 각 환자의 마지막 상태를 기억하기 위한 변수 (서버 실행 시 초기화)
# 예: {5: 'present', 6: 'sleeping'}
patient_states = {}

def on_connect(client, userdata, flags, rc):
    """브로커 연결 성공 시 실행되는 콜백 함수"""
    if rc == 0:
        print(f"✅ MQTT 브로커({MQTT_BROKER_IP})에 연결되었습니다.")
        client.subscribe(MQTT_TOPIC)
        print(f"   '{MQTT_TOPIC}' 토픽 구독을 시작합니다.")
    else:
        print(f"❌ 브로커 연결 실패! (코드: {rc})")

def on_message(client, userdata, msg):
    """
    메시지 수신 시 데이터를 각각 다른 테이블에 저장하는 메인 콜백 함수
    - smartring_logs: spo2, heartrate 데이터를 항상 저장
    - events: status 데이터는 상태가 변경되었을 때만 저장
    """
    topic = msg.topic
    payload = msg.payload.decode('utf-8')
    print(f"📨 메시지 수신 [토픽: {topic}] > {payload}")

    db_conn = None
    cursor = None
    try:
        # --- 1. 데이터 추출 및 파싱 ---
        patient_id = int(topic.split('/')[-1])
        data = json.loads(payload)
        
        timestamp_str = data.get("timestamp", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        spo2 = data.get("spo2")
        heartrate = data.get("heartrate")
        # JSON 페이로드에서 'status' 값을 직접 가져옵니다.
        status_from_payload = data.get("status")

        # --- 2. DB 연결 ---
        db_conn = mysql.connector.connect(
            host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME
        )
        cursor = db_conn.cursor()

        # --- 3. [작업 1] `smartring_logs` 테이블에 항상 데이터 저장 ---
        sql_log = "INSERT INTO smartring_logs (patient_id, timestamp, spo2, heartrate) VALUES (%s, %s, %s, %s)"
        values_log = (patient_id, timestamp_str, spo2, heartrate)
        cursor.execute(sql_log, values_log)
        print(f"   -> [로그 저장] 환자 ID {patient_id}의 스마트링 데이터 저장 완료.")

        # --- 4. [작업 2] `events` 테이블에는 상태가 변경되었을 때만 저장 ---
        if status_from_payload: # 'status' 필드가 메시지에 포함된 경우에만 실행
            last_known_status = patient_states.get(patient_id)

            if status_from_payload != last_known_status:
                print(f"   ✨ [상태 변경 감지] 환자 ID {patient_id}: '{last_known_status}' -> '{status_from_payload}'")
                
                sql_event = "INSERT INTO events (patient_id, event_type, event_timestamp) VALUES (%s, %s, %s)"
                values_event = (patient_id, status_from_payload, timestamp_str)
                cursor.execute(sql_event, values_event)
                
                # 새로운 상태를 메모리에 기억합니다.
                patient_states[patient_id] = status_from_payload
                print(f"   -> [이벤트 생성] 환자 ID {patient_id}의 '{status_from_payload}' 이벤트 생성 완료.")
            else:
                 print(f"   -> [상태 유지] 환자 ID {patient_id}는 '{status_from_payload}' 상태를 유지중입니다.")

        # --- 5. 모든 DB 작업 최종 확정 (Commit) ---
        db_conn.commit()
        print(f"💾 모든 변경사항이 데이터베이스에 최종 저장되었습니다.")

    except json.JSONDecodeError:
        print(f"❌ JSON 파싱 실패: {payload}")
    except (mysql.connector.Error, ValueError) as err:
        print(f"❌ 데이터베이스 오류 또는 데이터 변환 오류: {err}")
    except Exception as e:
        print(f"❌ 알 수 없는 오류 발생: {e}")
    finally:
        if cursor: cursor.close()
        if db_conn and db_conn.is_connected(): db_conn.close()

def run_mqtt_client():
    """MQTT 클라이언트를 실행하는 메인 함수"""
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

