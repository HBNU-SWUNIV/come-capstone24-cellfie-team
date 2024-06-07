import React from "react";
import { NavLink } from "react-router-dom";
import "../styles/common/Nav.css";

const Nav = () => {
  return (
    <nav>
      <div>
        <NavLink to="/">전체</NavLink>
      </div>
      <div>
        <NavLink to="/water1">유입수</NavLink>
      </div>
      <div>
        <NavLink to="/water2">수조1</NavLink>
      </div>
      <div>
        <NavLink to="/water3">수조2</NavLink>
      </div>
    </nav>
  );
};

export default Nav;
