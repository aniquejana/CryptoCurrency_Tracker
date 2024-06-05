import React from 'react';
import {
  AppBar,
  Container,
  MenuItem,
  Select,
  Toolbar,
  Typography,
} from "@material-ui/core";
import {
  createTheme,
  makeStyles,
  ThemeProvider,
} from "@material-ui/core/styles";
import { useHistory } from "react-router-dom";
import { CryptoState } from "../CryptoContext";
import AuthModal from "./Authentication/AuthModal";
import UserSidebar from './Authentication/UserSidebar';


const darkTheme = createTheme({
  palette: {
    primary: {
      main: "#fff",
    },
    type: "dark",
  },
});

const useStyles = makeStyles((theme) => ({
  title: {
    flex: 1,
    color: "#9A8174",
    fontFamily: "Montserrat",
    fontWeight: "bold",
    cursor: "pointer",
    textAlign: "center",
    fontSize: "35px",
  },
  appBar: {
    background: "linear-gradient(to right, #000000, #434343)", // Dark linear gradient background
  },
}));

const Header = () => {
  const classes = useStyles();
  const { currency, setCurrency, user } = CryptoState();
  const history = useHistory();

  return (
    <ThemeProvider theme={darkTheme}>
      <AppBar className={classes.appBar} position="static">
        <Container>
          <Toolbar>
            <Typography
              onClick={() => history.push(`/`)}
              variant="h6"
              className={classes.title}
            >
              COIN PIRATES
            </Typography>
            <Select
              variant="outlined"
              labelId="demo-simple-select-label"
              id="demo-simple-select"
              value={currency}
              style={{ width: 100, height: 40, marginLeft: 15, marginRight: 15 }}
              onChange={(e) => setCurrency(e.target.value)}
            >
              <MenuItem value={"USD"}>Dollar</MenuItem>
              <MenuItem value={"INR"}>Rupees</MenuItem>
            </Select>
            {user ? <UserSidebar /> : <AuthModal />}
          </Toolbar>
        </Container>
      </AppBar>
     
    </ThemeProvider>
  );
};

export default Header;