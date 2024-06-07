import axios from "axios";
import { useEffect, useState } from "react";

import "../../../../styles/chart/DataTextBox.css";
import { RotatingLines } from "react-loader-spinner";

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL;

const DataTextBox3 = () => {
  const [dataPoints, setDataPoints] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  //   const chartRef = useRef(null);

  const fetchData = async (tankId) => {
    setLoading(true);
    setError(null);
    try {
      const response_data = await axios.get(
        `${API_BASE_URL}/waterone?tankid=${tankId}`
      );
      const response_data_pred = await axios.get(
        `${API_BASE_URL}/pdo?tankid=${tankId}`
      );
      const combDataPoints = [response_data.data, response_data_pred.data[0]];

      setDataPoints(combDataPoints); // API로부터 데이터 받기
    } catch (error) {
      setError("데이터를 불러오는 데 실패했습니다.");
      console.error(error);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchData("rt2");
  }, []);
  return (
    <div>
      {loading && (
        <RotatingLines
          visible={true}
          height="96"
          width="96"
          color="grey"
          strokeWidth="5"
          animationDuration="0.75"
          ariaLabel="rotating-lines-loading"
          wrapperStyle={{}}
          wrapperClass=""
        />
      )}
      {error && <p>{error}</p>}
      {dataPoints ? (
        <div className="textBody">
          <p>시각: {formatTime(dataPoints[0].time)}</p>
          <p>수온: {dataPoints[0].wt} °C</p>
          <p>pH농도: {dataPoints[0].ph}</p>
          <p>염도: {dataPoints[0].sa} psu</p>
          <p>용존산소량: {dataPoints[0].wdo} ㎎/L</p>
          <p>용존산소량 예측값(30분 뒤): {dataPoints[1].pdo} ㎎/L</p>
        </div>
      ) : (
        <p>데이터가 없습니다.</p>
      )}
    </div>
  );
};

const formatTime = (time) => {
  if (!time) return "";
  // 예상 입력 형식: "202404230956"
  const year = time.substring(0, 4);
  const month = time.substring(4, 6);
  const day = time.substring(6, 8);
  const hour = time.substring(8, 10);
  const minute = time.substring(10, 12);
  return `${year}-${month}-${day} ${hour}:${minute}`;
};

export default DataTextBox3;
