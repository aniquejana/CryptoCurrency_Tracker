import React from "react";
//import Display from "../components/Display/DisplayCoin";
import CoinsTable from "../components/CoinsTable";
import News from "../components/News"
import Footer from "../components/Footer";



const Homepage = () => {
  return (
    <>
      <News />
      <CoinsTable />
      <Footer />
    </>
  );
};

export default Homepage;
