import React, { useRef, useState, useEffect } from "react";
import { Line } from "react-chartjs-2";
import axios from "axios";
import "chartjs-adapter-date-fns";
import { ko } from "date-fns/locale";
import { RotatingLines } from "react-loader-spinner";
import { Chart, registerables } from "chart.js";

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL;

Chart.register(...registerables);

const W3WPDO = () => {
  const [chartData, setChartData] = useState({ datasets: [] });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const chartRef = useRef(null);

  const fetchChartData = async (tankId) => {
    setLoading(true);
    setError(null);
    try {
      const response_pred = await axios.get(
        `${API_BASE_URL}/pdoweek?tankid=${tankId}`
      );
      const dataPointPred = formatPredictionData(response_pred.data);

      // 데이터 포맷팅

      setChartData({
        datasets: [
          {
            label: "용존산소 예측값",
            data: dataPointPred, // 두 번째 데이터셋
            backgroundColor: "#FF0000",
            borderColor: "#FF0000",
          },
        ],
      });
    } catch (error) {
      setError("데이터를 불러오는 데 실패했습니다.");
      console.error(error);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchChartData("rt2");
    const chart = chartRef.current;
    return () => {
      chart?.destroy();
    };
  }, []);

  return (
    <div>
      {loading ? (
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
      ) : error ? (
        <p>{error}</p>
      ) : (
        <Line data={chartData} options={options} height={200} width={400} />
      )}
    </div>
  );
};

export default W3WPDO;

function formatPredictionData(predictionData) {
  // 예측 데이터 포맷팅
  const filteredDataPoints = predictionData.filter((dp) => {
    const date = parseDate(dp.time);
    return date.getMinutes() === 0 || date.getMinutes() === 30;
  });
  const predDO = filteredDataPoints.map((pred) => ({
    x: parseDate(pred.time),
    y: pred.pdoweek,
  }));

  return predDO;
}

function parseDate(timeString) {
  const year = parseInt(timeString.substring(0, 4), 10);
  const month = parseInt(timeString.substring(4, 6), 10) - 1;
  const day = parseInt(timeString.substring(6, 8), 10);
  const hour = parseInt(timeString.substring(8, 10), 10);
  const minute = parseInt(timeString.substring(10, 12), 10);

  return new Date(year, month, day, hour, minute);
}

const options = {
  responsive: true,
  interaction: {
    intersect: false,
  },
  scales: {
    x: {
      grid: {
        display: false,
      },
      type: "time",
      time: {
        unit: "day",
        tooltipFormat: "MMM dd",
        displayFormats: {
          day: "MMM dd",
        },
      },
      adapters: {
        date: {
          locale: ko,
        },
      },
      max: new Date().setDate(new Date().getDate() + 8),
    },
    y: {},
  },
  plugins: {
    legend: {
      position: "bottom",
    },
  },
};
