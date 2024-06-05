import { AppBar,makeStyles } from "@material-ui/core";
import { BrowserRouter, Route } from "react-router-dom";
import Homepage from "./Pages/HomePage";
import CoinPage from "./Pages/CoinPage";
import Header from "./components/Header";
import "./App.css";
import '@fortawesome/fontawesome-free/css/all.min.css';
import Alert from "./components/Alert";
import Watchlist from "./Pages/Watchlist";
import Profile from "./Pages/Profile";

const useStyles = makeStyles(() => ({
  App: {
    backgroundColor: "#050206",
    color: "white",
    minHeight: "100vh",
  },
}));

function App() {
  const classes = useStyles();

  return (
    <BrowserRouter>
      <div className={classes.App}>
        <Header />
        <Route path="/" component={Homepage} exact />
        <Route path="/coins/:id" component={CoinPage} exact />
        <Route path="/watchlist" component={Watchlist} exact />
        <Route path="/profile" component={Profile} exact />
        
        
        
      </div>
      <Alert />
    </BrowserRouter>
  );
}

export default App;