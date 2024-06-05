import React, { useState } from "react";
import { makeStyles } from "@material-ui/core/styles";
import { Avatar, Typography, Paper, TextField, Button } from "@material-ui/core";
import { CryptoState } from "../CryptoContext";

const useStyles = makeStyles((theme) => ({
  container: {
    display: "flex",
    marginTop: theme.spacing(6),
    marginLeft: theme.spacing(3),
  },
  profile: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: theme.spacing(2),
  },
  avatar: {
    width: 150,
    height: 150,
    backgroundColor: "#EEBC1D",
  },
  userName: {
    fontSize: 25,
    fontWeight: "bold",
  },
  detailsContainer: {
    display: "flex",
    flexDirection: "column",
    gap: theme.spacing(2),
    marginLeft: theme.spacing(4),
    width: "80%",
    padding: theme.spacing(2),
  },
  detailBox: {
    padding: theme.spacing(2),
    backgroundColor: theme.palette.background.paper,
    borderRadius: theme.shape.borderRadius,
    boxShadow: theme.shadows[1],
  },
  detailLabel: {
    fontWeight: "bold",
    marginBottom: theme.spacing(1),
  },
  detailValue: {
    wordWrap: "break-word",
  },
  buttonContainer: {
    display: "flex",
    justifyContent: "flex-end",
    marginTop: theme.spacing(2),
  },
  button: {
    marginLeft: theme.spacing(1),
  },
}));

export default function Profile() {
  const classes = useStyles();
  const { user } = CryptoState();
  
  const [editMode, setEditMode] = useState(false);
  const [location, setLocation] = useState("");
  const [phoneNumber, setPhoneNumber] = useState("");
  const [accountNumber, setAccountNumber] = useState("");

  const handleEditClick = () => setEditMode(true);
  const handleSaveClick = () => setEditMode(false);

  return (
    <div className={classes.container}>
      <div className={classes.profile}>
        <Avatar
          src={user.photoURL}
          alt={user.displayName || user.email}
          className={classes.avatar}
        />
        <Typography className={classes.userName}>
          {user.displayName || "User"}
        </Typography>
      </div>
      <div className={classes.detailsContainer}>
        <Paper className={classes.detailBox}>
          <Typography className={classes.detailLabel}>Email:</Typography>
          <Typography className={classes.detailValue}>{user.email}</Typography>
        </Paper>
        <Paper className={classes.detailBox}>
          <Typography className={classes.detailLabel}>Location:</Typography>
          {editMode ? (
            <TextField
              fullWidth
              variant="outlined"
              value={location}
              onChange={(e) => setLocation(e.target.value)}
              placeholder="Enter your location"
            />
          ) : (
            <Typography className={classes.detailValue}>{location || "User's Location"}</Typography>
          )}
        </Paper>
        <Paper className={classes.detailBox}>
          <Typography className={classes.detailLabel}>Phone:</Typography>
          {editMode ? (
            <TextField
              fullWidth
              variant="outlined"
              value={phoneNumber}
              onChange={(e) => setPhoneNumber(e.target.value)}
              placeholder="Enter your phone number"
            />
          ) : (
            <Typography className={classes.detailValue}>{phoneNumber || "User's Phone Number"}</Typography>
          )}
        </Paper>
        <Paper className={classes.detailBox}>
          <Typography className={classes.detailLabel}>Account Number:</Typography>
          {editMode ? (
            <TextField
              fullWidth
              variant="outlined"
              value={accountNumber}
              onChange={(e) => setAccountNumber(e.target.value)}
              placeholder="Enter your account number"
            />
          ) : (
            <Typography className={classes.detailValue}>{accountNumber || "User's Account Number"}</Typography>
          )}
        </Paper>
        <div className={classes.buttonContainer}>
          {editMode ? (
            <Button variant="contained" color="primary" onClick={handleSaveClick} className={classes.button}>
              Save
            </Button>
          ) : (
            <Button variant="contained" color="secondary" onClick={handleEditClick} className={classes.button}>
              Edit
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}