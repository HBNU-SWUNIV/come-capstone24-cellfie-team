import React from "react";
import W3DO from "../components/charts/charts-division/water1/W1DO";
import W3PH from "../components/charts/charts-division/water1/W1PH";
import W3WT from "../components/charts/charts-division/water1/W1WT";
import W3SALT from "../components/charts/charts-division/water3/W3SALT";
import Button from "../components/Button";
import DataTextBox3 from "../components/charts/charts-division/water3/DataTextBox3";

import "../styles/chart/Water1.css";
import W3WPDO from "../components/charts/charts-division/water3/W3WPDO";

class Water3 extends React.Component {
  render() {
    return (
      <div className="body">
        <Button />
        <div className="dataBox">
          <DataTextBox3 />
        </div>
        <div className="dataCharts">
          <div className="dataChart">
            <div className="dataName">용존산소</div>
            <W3DO />
          </div>
          <div className="dataChart">
            <div className="dataName">용존산소</div>
            <W3WPDO />
          </div>
          <div className="dataChart">
            <div className="dataName">pH농도</div>
            <W3PH />
          </div>
        </div>
        <div className="dataCharts">
          <div className="dataChart">
            <div className="dataName">수온</div>
            <W3WT />
          </div>
          <div className="dataChart">
            <div className="dataName">염도</div>
            <W3SALT />
          </div>
        </div>
      </div>
    );
  }
}

export default Water3;
