import express, { Request, Response, NextFunction } from "express";
import axios from "axios";
import cors from 'cors';

const app = express();
const port = 3100;

app.listen(port, () => {
  console.log(`Backend server is alive on port: ${port}.`);
});

app.use(cors({
    origin: 'http://localhost:5173',
}))

const vaderSI = (request: Request, response: Response) => {
  axios
    .get(
      "http://localhost:3200/sentimentIntensity?" +
        request.originalUrl.split("%20").join(" ")
    )
    .then((vader_response) => {
      response.set('Access-Control-Allow-Origin', '*')
      response.status(200).send(vader_response.data);
    });

};

app.get("/vader", vaderSI);
