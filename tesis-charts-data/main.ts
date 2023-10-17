const defaultEdibleFilm = [
  0.90,
  0.22,
  0.19,
  0.00,
  1.34,
  0.00,
  96.98,
  0.37,
  60.00,
  50.00,
  24.00,
];

if (import.meta.main) {
  // const data = generateData();
  // saveDataForModel("ts", data);
  // saveDataForModel("wvp", data);
  const data = [[
    0.90,
    0.22,
    0.19,
    0.00,
    1.34,
    0.00,
    96.98,
    0.37,
    60.00,
    50.00,
    24.00,
  ]];

  const films: number[][] = [];

  for (let i = 0; i < 10; i++) {
    const film = [...defaultEdibleFilm];
    film[9] = i * 10;
    films.push(film);
  }

  console.log({ films });
  const predictions = await fetchPredictions("ts", films);
  console.log({ predictions });
}

async function saveDataForModel(
  model: string,
  data: number[][],
): Promise<void> {
  const predictions = await fetchPredictions(model, data);
  savePredictionsToCSV(model, predictions);
}

function savePredictionsToCSV(model: string, predictions: number[][]): void {
  createCSVHeader(model);
  for (const edibleFilm of predictions) {
    const [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, y] = edibleFilm;
    Deno.writeTextFileSync(
      `${model}_predictions.csv`,
      `\n${x1},${x2},${x3},${x4},${x5},${x6},${x7},${x8},${x9},${x10},${x11},${y}`,
      { append: true },
    );
  }
}

function createCSVHeader(model: string): void {
  Deno.writeTextFileSync(
    `${model}_predictions.csv`,
    `%Chi,%Gel,%Gly,%Pec,%Sta,%Oil,%W,%AA,T(°C),%RH,t(h),${model}`,
    { append: true },
  );
}

async function fetchPredictions(
  model: string,
  edibleFilm: number[][],
): Promise<number[][]> {
  const response = await makeRequestToModel(model, edibleFilm);
  const { data: predictions } = await response.json();
  return predictions;
}

function makeRequestToModel(
  model: string,
  edibleFilm: number[][],
): Promise<Response> {
  const baseUrl = `http://localhost:8000/model/${model}`;
  return fetch(baseUrl, {
    method: "POST",
    body: JSON.stringify(edibleFilm),
    headers: {
      "Content-Type": "application/json",
    },
  });
}

function generateData(): number[][] {
  const data: number[][] = [];

  for (let i = 0; i < defaultEdibleFilm.length; i++) {
    for (let j = 0; j < 10; j++) {
      const edibleFilm = [...defaultEdibleFilm];
      edibleFilm[i] += j + 1;
      data.push(edibleFilm);
    }
  }

  return data;
}
