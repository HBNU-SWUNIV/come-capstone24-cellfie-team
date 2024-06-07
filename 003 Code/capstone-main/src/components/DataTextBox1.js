import axios from "axios";
import { useEffect, useState } from "react";

import "../styles/chart/DataTextBox.css";
import { RotatingLines } from "react-loader-spinner";

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL;

const DataTextBox1 = () => {
  const [dataPoints, setDataPoints] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  //   const chartRef = useRef(null);

  const fetchData = async (tankId) => {
    setLoading(true);
    setError(null);
    try {
      const response_data = await axios.get(
        `${API_BASE_URL}/test?tankid=${tankId}`
      );

      const dataPoint = response_data.data[0];

      const response_pred = await axios.get(
        `${API_BASE_URL}/pdo?tankid=${tankId}`
      );
      const dataPointPred = response_pred.data[0];

      const dataPointsComb = { ...dataPoint, dataPointPred };

      setDataPoints(dataPointsComb); // API로부터 데이터 받기
      console.log(dataPointsComb);
    } catch (error) {
      setError("데이터를 불러오는 데 실패했습니다.");
      console.error(error);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchData("iw1");
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
        <div className="textBodyOverview">
          <p>시각: {formatTime(dataPoints.time)}</p>
          <p>수온: {dataPoints.wt}</p>
          <p>pH농도: {dataPoints.ph}</p>
          <p>염도: {dataPoints.sa}</p>
          <p>용존산소량: {dataPoints.wdo}</p>
          <p>용존산소량 예측값: {dataPoints.pdo}</p>
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

export default DataTextBox1;
