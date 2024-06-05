import axios from "axios";
import { useEffect, useState } from "react";
import { HistoricalChart } from "../config/api";
import { Line } from "react-chartjs-2";
import {
  CircularProgress,
  createTheme,
  makeStyles,
  ThemeProvider,
  Button,
  LinearProgress,
} from "@material-ui/core";
import SelectButton from "./SelectButton";
import { chartDays } from "../config/data";
import { CryptoState } from "../CryptoContext";

const useStyles = makeStyles((theme) => ({
  container: {
    width: "75%",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    marginTop: 25,
    padding: 40,
    [theme.breakpoints.down("md")]: {
      width: "100%",
      marginTop: 0,
      padding: 20,
      paddingTop: 0,
    },
  },
  predictionContainer: {
    marginTop: 20,
    padding: 20,
    borderRadius: 10,
    backgroundColor: "black",
    width: "100%",
    textAlign: "center",
  },
  predictionText: {
    fontSize: "1.2rem",
    fontWeight: "bold",
  },
  buttonContainer: {
    display: "flex",
    justifyContent: "flex-end",
    width: "100%",
  },
  progressBar: {
    width: "100%",
    marginTop: 20,
  },
}));

const CoinInfo = ({ coin }) => {
  const [historicData, setHistoricData] = useState();
  const [days, setDays] = useState(1);
  const { currency } = CryptoState();
  const [flag, setFlag] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const classes = useStyles();

  const fetchHistoricData = async () => {
    const { data } = await axios.get(HistoricalChart(coin.id, days, currency));
    setFlag(true);
    setHistoricData(data.prices);
  };

  useEffect(() => {
    fetchHistoricData();
  }, [days]);

  const convertToCSV = () => {
    if (!historicData || !coin.name) return;

    let csvContent = `Date,Time,Price (${currency})\n`;
    historicData.forEach((dataPoint) => {
      const date = new Date(dataPoint[0]);
      const time = date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      const value = dataPoint[1];
      csvContent += `${date.toISOString().slice(0, 10)},${time},${value}\n`;
    });

    return csvContent;
  };

  const predict = async () => {
    try {
      setIsLoading(true);

      const csvData = convertToCSV();
      console.log("CSV Data:", csvData);

      const response = await axios.post("http://127.0.0.1:5000/predict", csvData, {
        headers: {
          "Content-Type": "text/csv",
        },
      });

      console.log("Backend Response:", response.data);

      setPrediction(response.data);
    } catch (error) {
      console.error("Error predicting:", error);
      if (error.response) {
        console.error("Response Data:", error.response.data);
        console.error("Status Text:", error.response.statusText);
        console.error("Status Code:", error.response.status);
      } else if (error.request) {
        console.error("No Response Received:", error.request);
      } else {
        console.error("Error:", error.message);
      }
    } finally {
      setIsLoading(false);
    }
  };

  const darkTheme = createTheme({
    palette: {
      primary: {
        main: "#fff",
      },
      type: "dark",
    },
  });

  const getRecommendation = (avgPrice, currentPrice) => {
    if (currentPrice < avgPrice) {
      return "Buy";
    } else if (currentPrice > avgPrice) {
      return "Sell";
    } else {
      return "Hold";
    }
  };

  const currentPrice = historicData && historicData[historicData.length - 1][1];

  return (
    <ThemeProvider theme={darkTheme}>
      <div className={classes.container}>
        {!historicData || !flag ? (
          <CircularProgress style={{ color: "#F7F1EA" }} size={250} thickness={1} />
        ) : (
          <>
            <Line
              data={{
                labels: historicData.map((coin) => {
                  let date = new Date(coin[0]);
                  let time = date.getHours() > 12
                    ? `${date.getHours() - 12}:${date.getMinutes()} PM`
                    : `${date.getHours()}:${date.getMinutes()} AM`;
                  return days === 1 ? time : date.toLocaleDateString();
                }),

                datasets: [
                  {
                    data: historicData.map((coin) => coin[1]),
                    label: `Price ( Past ${days} Days ) in ${currency}`,
                    borderColor: "#9A8174",
                  },
                ],
              }}
              options={{
                elements: {
                  point: {
                    radius: 1,
                  },
                },
              }}
            />
            <div
              style={{
                display: "flex",
                marginTop: 20,
                justifyContent: "space-around",
                width: "100%",
              }}
            >
              {chartDays.map((day) => (
                <SelectButton
                  key={day.value}
                  onClick={() => {
                    setDays(day.value);
                    setFlag(false);
                  }}
                  selected={day.value === days}
                >
                  {day.label}
                </SelectButton>
              ))}
            </div>
            <div className={classes.buttonContainer}>
              <Button
                onClick={predict}
                variant="contained"
                color="primary"
                style={{ marginTop: 20, fontSize: "1.2rem" }}
                disabled={isLoading}
              >
                {isLoading ? "Predicting..." : "Predict"}
              </Button>
            </div>
            {isLoading && <LinearProgress className={classes.progressBar} />}
            {prediction && (
              <div className={classes.predictionContainer}>
                <p className={classes.predictionText}>
                  Predicted Future Price Range: {prediction.future_price_range?.[0]?.toFixed(2) || "N/A"} -{" "}
                  {prediction.future_price_range?.[1]?.toFixed(2) || "N/A"}
                </p>
                <p className={classes.predictionText}>
                  Recommendation: {getRecommendation(prediction.average_future_price, currentPrice)}
                </p>
                {prediction.accuracy_percentage && (
                  <p className={classes.predictionText}>
                    Accuracy: {prediction.accuracy_percentage.toFixed(2)}%
                  </p>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </ThemeProvider>
  );
};

export default CoinInfo;
