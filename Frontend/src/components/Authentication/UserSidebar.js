import React, { useState } from "react";
import { makeStyles } from "@material-ui/core/styles";
import { Avatar, button, Drawer, List, ListItem, ListItemText } from "@material-ui/core";
import { CryptoState } from "../../CryptoContext";
import { signOut } from "firebase/auth";
import { auth } from "../../firebase";
import { useHistory } from "react-router-dom";

const drawerWidth = 200;

const useStyles = makeStyles((theme) => ({
  container: {
    padding: "20px",
    height: "100%",
    display: "flex",
    flexDirection: "column",
    fontFamily: "Arial, sans-serif",
  },
  navbar: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "20px",
  },
  profile: {
    display: "flex",
    alignItems: "center",
    gap: "20px",
  },
  logo: {
    fontSize: "24px",
    fontWeight: "bold",
    cursor: "pointer",
  },
  userIcon: {
    cursor: "pointer",
  },
  logout: {
    backgroundColor: "#f9977b",
    color: "white",
    "&:hover": {
      backgroundColor: "#f27a49",
    },
  },
  drawer: {
    width: drawerWidth,
  },
}));

export default function UserSideBar() {
  const classes = useStyles();
  const { user, setAlert } = CryptoState();
  const history = useHistory();
  const [drawerOpen, setDrawerOpen] = useState(false);

  const logOut = () => {
    signOut(auth);
    setAlert({
      open: true,
      type: "success",
      message: "Logout Successful!",
    });
  };

  const navigateToWatchlist = () => {
    history.push("/watchlist");
  };

  const toggleDrawer = () => {
    setDrawerOpen(!drawerOpen);
  };

  return (
    <div className={classes.container}>
      <div className={classes.navbar}>
        <div className={classes.profile}>
          <Avatar
            src={user.photoURL}
            alt={user.displayName || user.email}
            className={classes.userIcon}
            onClick={toggleDrawer}
          />
        </div>
      </div>
      <Drawer anchor="right" open={drawerOpen} onClose={toggleDrawer} className={classes.drawer}>
        <List>
          <ListItem button onClick={logOut}>
            <ListItemText primary="Log-Out" />
          </ListItem>
          <ListItem button onClick={navigateToWatchlist}>
            <ListItemText primary="Watchlist" />
          </ListItem>
          <ListItem button onClick={() => history.push("/profile")}>
            <ListItemText primary="Profile" />
          </ListItem>
        </List>
      </Drawer>
    </div>
  );
}
