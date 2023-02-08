import PageContainer from '../common/pageContainer'
import { vaderResponse } from '../interfaces/vaderResponse';
import Grid from '@mui/material/Unstable_Grid2'
import TextField from '@mui/material/TextField';
import { useState } from 'react';
import Typography from '@mui/material/Typography';
import CardContent from '@mui/material/CardContent';
import Card from '@mui/material/Card';
import { Stack } from '@mui/system';

export default function VADERpage() {
  const [vaderResponse, updateResponse] = useState({
    overall: "Please",
    positivity: "Enter",
    negativity: "Some",
    neutrality: "Data"
  });
  const [vaderMessage, updateVaderMessage] = useState(
    "Enter some text in the box to the left"
  );

  async function userInput(value: string) {
    const response = await fetch("http://localhost:3100/vader?=" + JSON.stringify(value), {
      method: 'GET',
      mode: 'cors',
      headers: { 'Content-Type': 'application/json' },
    })

    const data = await response.json()
    const vader: vaderResponse = {
      overall: data.compound,
      positivity: data.pos,
      negativity: data.neg,
      neutrality: data.neu
    }

    if (parseFloat(data.compound) > 0) {
      updateVaderMessage("VADER considers your message positive")
    } else if (parseFloat(data.compound) < 0) {
      updateVaderMessage("VADER considers your message negative")
    } else if (parseFloat(data.compound) == 0) {
      updateVaderMessage("VADER considers your message neutral")
    }

    updateResponse(vader)
  }

  return (
    <PageContainer>
      <Grid container spacing={2}>

        <Grid xs={7}>
          <TextField
            id="outlined-textarea"
            label="Put the text you want to input here:"
            placeholder="I really did not like that dog..."
            rows={25}
            variant="filled"
            fullWidth
            multiline
            onChange={(e) => {
              userInput(e.target.value)
            }}
          />
        </Grid>

        <Grid xs={5}>
          <Stack spacing={2}>

            <Card>
              <CardContent>
                <Typography variant="h6" component="div">
                  VADER Output:
                </Typography>
                <Typography sx={{ mb: 1.5 }} color="text.secondary">
                  What does VADER think about your text?
                </Typography>
                <Typography variant="body2">
                  Overall: <b>{vaderResponse.overall}</b>
                  <br />
                  Positivity: <b>{vaderResponse.positivity}</b>
                  <br />
                  Negativity: <b>{vaderResponse.negativity}</b>
                  <br />
                  Neautrality: <b>{vaderResponse.neutrality}</b>
                </Typography>
                <Typography sx={{ mt: 1.5 }} color="text.secondary">
                  {vaderMessage}
                </Typography>
              </CardContent>
            </Card>

            <Card sx={{ minWidth: "auto"}}>
              <CardContent>
                <Typography variant="h6" component="div">
                  What is VADER?
                </Typography>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  VADER (Valence Aware Dictionary for sEntiment Reasoning)
                  is a model used for text sentiment analysis that is
                  sensitive to both polarity (positive/negative)
                  and intensity (strength) of emotion.
                  Introduced in 2014, VADER text sentiment analysis uses
                  a human-centric approach, combining qualitative
                  analysis and empirical validation by using human raters
                  and the wisdom of the crowd.
                </Typography>

              </CardContent>
            </Card>
          </Stack>
        </Grid>

      </Grid>
    </PageContainer>
  )
}