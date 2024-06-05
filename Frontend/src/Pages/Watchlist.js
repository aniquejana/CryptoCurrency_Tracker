import React from "react";
import { makeStyles } from "@material-ui/core/styles";
import { Avatar } from "@material-ui/core";
import { CryptoState } from "../CryptoContext";
import { signOut } from "firebase/auth";
import { auth, db } from "../firebase";
import { numberWithCommas } from "../components/CoinsTable";
import { AiFillDelete } from "react-icons/ai";
import { doc, setDoc } from "firebase/firestore";
import { useHistory } from "react-router-dom"; // Import useHistory hook

const useStyles = makeStyles({
  container: {
    width: "100%",
    padding: 15,
    fontFamily: "monospace",
  },
  profile: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: "20px",
    marginBottom: 20,
  },
  watchlist: {
    width: "100%",
    borderCollapse: "collapse",
  },
  coinRow: {
    backgroundColor: "#16171a",
    cursor: "pointer",
    "&:hover": {
      backgroundColor: "#131111",
    },
    fontFamily: "Montserrat",
  },
  coinHeader: {
    backgroundColor: "#F9977B",
    fontWeight: "700",
    color: "black",
    fontFamily: "Montserrat",
    padding: "10px 15px",
    textAlign: "left",
  },
  coinData: {
    padding: "10px 15px",
    textAlign: "left",
  },
  coinImage: {
    maxWidth: 40,
    marginRight: 10,
  },
  coinSymbol: {
    fontWeight: 700,
    color: "#ffffff",
    marginLeft: 45,
    fontSize: 20,
    textTransform: "uppercase",
  },
});

export default function Watchlist() {
  const classes = useStyles();
  const history = useHistory(); // Initialize useHistory hook
  const { user, setAlert, watchlist, coins, symbol } = CryptoState();

  const logOut = () => {
    signOut(auth);
    setAlert({
      open: true,
      type: "success",
      message: "Logout Successful!",
    });
  };

  const removeFromWatchlist = async (coin) => {
    const coinRef = doc(db, "watchlist", user.uid);
    try {
      await setDoc(
        coinRef,
        { coins: watchlist.filter((wish) => wish !== coin?.id) },
        { merge: true }
      );

      setAlert({
        open: true,
        message: `${coin.name} Removed from the Watchlist!`,
        type: "success",
      });
    } catch (error) {
      setAlert({
        open: true,
        message: error.message,
        type: "error",
      });
    }
  };

  const handleCoinClick = (coinId) => {
    history.push(`/coins/${coinId}`); // Navigate to the coin page
  };

  return (
    <div className={classes.container}>
      <div className={classes.profile}>
        <Avatar
          style={{
            height: 100,
            width: 100,
            cursor: "pointer",
            backgroundColor: "#EEBC1D",
          }}
          src={user.photoURL}
          alt={user.displayName || user.email}
        />
        <span style={{ fontSize: 25, fontWeight: "bold" }}>
          {user.displayName || user.email}
        </span>
      </div>
      <table className={classes.watchlist}>
        <thead>
          <tr>
            <th className={classes.coinHeader}>Name</th>
            <th className={classes.coinHeader}>Price</th>
            <th className={classes.coinHeader}>24hr Change</th>
            <th className={classes.coinHeader}>Market Cap</th>
            <th className={classes.coinHeader}></th>
          </tr>
        </thead>
        <tbody>
          {coins.map((coin) => (
            watchlist.includes(coin.id) && (
              <tr
                key={coin.id}
                className={classes.coinRow}
                onClick={() => handleCoinClick(coin.id)} // Add onClick handler
              >
                <td className={classes.coinData}>
                  {coin.symbol && (
                    <span className={classes.coinSymbol}>{coin.symbol}</span>
                  )}
                  <br />
                  <img
                    src={coin.image}
                    alt={coin.name}
                    className={classes.coinImage}
                  />
                  {coin.name}
                </td>
                <td className={classes.coinData}>
                  {symbol} {numberWithCommas(coin.current_price.toFixed(2))}
                </td>
                <td className={classes.coinData}>
                  {coin.price_change_percentage_24h > 0 ? (
                    <span style={{ color: "green" }}>
                      +{coin.price_change_percentage_24h.toFixed(2)}%
                    </span>
                  ) : (
                    <span style={{ color: "red" }}>
                      {coin.price_change_percentage_24h.toFixed(2)}%
                    </span>
                  )}
                </td>
                <td className={classes.coinData}>
                  {symbol} {numberWithCommas(coin.market_cap.toFixed(0))}
                </td>
                <td className={classes.coinData}>
                  <AiFillDelete
                    style={{ cursor: "pointer" }}
                    fontSize="16"
                    onClick={() => removeFromWatchlist(coin)}
                  />
                </td>
              </tr>
            )
          ))}
        </tbody>
      </table>
      {/* Uncomment if Log Out button is needed */}
      {/* <Button
        variant="contained"
        color="secondary"
        onClick={logOut}
        style={{ marginTop: 20 }}
      >
        Log Out
      </Button> */}
    </div>
  );
}
