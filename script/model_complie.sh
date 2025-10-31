#!/bin/bash

# --- 설정 변수 ---
BASE_JSON_PATH="./base_model_config.json"
ONNX_MODEL_PATH="./yolo_pose_model.onnx"
DX_COM_EXECUTABLE="./dx_com/dx_com"
OUTPUT_BASE_DIR="./quantization_results"
LOG_FILE="${OUTPUT_BASE_DIR}/compilation_times.log"

# --- 실험할 파라미터 조합 ---
CALIBRATION_METHODS=("ema" "minmax")
CALIBRATION_NUMS=(1 10 50 100 500 1000)
INTERPOLATION_MODES=("LINEAR" "CUBIC")

cat > ${BASE_JSON_PATH} << EOL
{
  "inputs": {
    "images": [1, 3, 640, 640]
  },
  "default_loader": {
    "dataset_path": "./image",
    "file_extensions": ["jpeg", "jpg", "png", "JPEG"],
    "preprocessings": [
      {
        "resize": {
          "width": 640,
          "height": 640
        }
      },
      {
        "convertColor": {
          "form": "BGR2RGB"
        }
      },
      {
        "div": {
          "x": 255.0
        }
      },
      {
        "transpose": {
          "axis": [2, 0, 1]
        }
      },
      {
        "expandDim": {
          "axis": 0
        }
      }
    ]
  },
  "calibration_num": 100,
  "calibration_method": "ema"
}
EOL

# --- 메인 실행 로직 ---
mkdir -p ${OUTPUT_BASE_DIR}
echo "--- DEEPX Model Compilation ---" > ${LOG_FILE}
echo "Start Time: $(date)" >> ${LOG_FILE}
echo "========================================" >> ${LOG_FILE}
echo "" >> ${LOG_FILE}

for method in "${CALIBRATION_METHODS[@]}"; do
  for num in "${CALIBRATION_NUMS[@]}"; do
    for interp in "${INTERPOLATION_MODES[@]}"; do

      output_name="yolo_pose_${method}_${num}_${interp}"
      output_dir="${OUTPUT_BASE_DIR}/${output_name}"
      temp_json_path="/tmp/temp_config_${output_name}.json"

      echo "Starting compilation for: ${output_name}"

      jq \
        --arg cm "$method" \
        --argjson cn "$num" \
        --arg interp "$interp" \
        '.calibration_method = $cm | .calibration_num = $cn | .default_loader.preprocessings[0].resize.interpolation = $interp' \
        ${BASE_JSON_PATH} > ${temp_json_path}

      start_time=$(date +%s)

      { time ${DX_COM_EXECUTABLE} \
        -m ${ONNX_MODEL_PATH} \
        -c ${temp_json_path} \
        -o ${output_dir} \
        --shrink > /dev/null; } 2> time_output.txt

      end_time=$(date +%s)
      elapsed_time=$((end_time - start_time))

      if [ $? -eq 0 ]; then
        status="SUCCESS"
        echo "SUCCESS: ${output_name} completed."
      else
        status="FAILED"
        echo "FAILED: ${output_name} failed."
      fi

      echo "Configuration: ${output_name}" >> ${LOG_FILE}
      echo "Status: ${status}" >> ${LOG_FILE}
      echo "Elapsed Time: ${elapsed_time} seconds" >> ${LOG_FILE}
      cat time_output.txt >> ${LOG_FILE}

    done
  done
done

rm time_output.txt