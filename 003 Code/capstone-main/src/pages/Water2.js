import React from "react";
import W2DO from "../components/charts/charts-division/water2/W2DO";
import W2PH from "../components/charts/charts-division/water2/W2PH";
import W2WT from "../components/charts/charts-division/water2/W2WT";
import W2SALT from "../components/charts/charts-division/water2/W2SALT";
import Button from "../components/Button";
import DataTextBox2 from "../components/charts/charts-division/water2/DataTextBox2";

import "../styles/chart/Water1.css";

class Water2 extends React.Component {
  render() {
    return (
      <div className="body">
        <Button />
        <div className="dataBox">
          <DataTextBox2 />
        </div>
        <div className="dataCharts">
          <div className="dataChart">
            <div className="dataName">용존산소</div>
            <W2DO />
          </div>
          <div className="dataChart">
            <div className="dataName">pH농도</div>
            <W2PH />
          </div>
        </div>
        <div className="dataCharts">
          <div className="dataChart">
            <div className="dataName">수온</div>
            <W2WT />
          </div>
          <div className="dataChart">
            <div className="dataName">염도</div>
            <W2SALT />
          </div>
        </div>
      </div>
    );
  }
}

export default Water2;
