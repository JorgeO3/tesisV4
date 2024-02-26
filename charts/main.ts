import { join } from "https://deno.land/std@0.206.0/path/join.ts";

const defaultEdibleFilm = [
  0.499,
  0.0,
  0.3,
  0.0,
  0.0,
  0.0,
  40,
  54,
  24,
];

if (import.meta.main) {
  const data = generateData();
  saveDataForModel("ts", data);
  saveDataForModel("wvp", data);
  saveDataForModel("e", data);
}

async function saveDataForModel(
  model: string,
  data: number[][],
): Promise<void> {
  let predictions = await fetchPredictions(model, data);
  predictions = predictions.map((pred, i) => [...data[i], ...pred]);
  savePredictionsToCSV(model, predictions);
}

function savePredictionsToCSV(model: string, predictions: number[][]): void {
  const dataDir = Deno.env.get("DATA_DIR");
  const filePath = join(dataDir!, `${model}_predictions.csv`);
  createCSVHeader(model, filePath);

  for (const pred of predictions) {
    const [x1, x2, x3, x4, x5, x6, x7, x8, x9, y] = pred;
    Deno.writeTextFileSync(
      filePath,
      `\n${x1},${x2},${x3},${x4},${x5},${x6},${x7},${x8},${x9},${y}`,
      { append: true },
    );
  }
}

function createCSVHeader(model: string, filePath: string): void {
  Deno.writeTextFileSync(
    filePath,
    `%Chi,%Gel,%Gly,%Pec,%Sta,%Oil,T(°C),%RH,t(h),${model}`,
    { append: true },
  );
}

async function fetchPredictions(
  model: string,
  edibleFilm: number[][],
): Promise<number[][]> {
  const response = await makeRequestToModel(model, edibleFilm);
  const predictions = await response.json();
  return predictions;
}

function makeRequestToModel(
  model: string,
  edibleFilm: number[][],
): Promise<Response> {
  const baseUrl = `http://localhost:8000/model`;
  const body = { model, payload: edibleFilm };

  return fetch(baseUrl, {
    method: "POST",
    body: JSON.stringify(body),
    headers: {
      "Content-Type": "application/json",
    },
  });
}

function generateData(): number[][] {
  const data: number[][] = [];
  // this value is important because it determines the base
  //value that the algorithm will use to perform the increments.
  const base = [1, 1.99, 0.89, 0.15, 0.69, 2.28, 31, 37.5, 169.25];

  for (let i = 0; i < defaultEdibleFilm.length; i++) {
    for (let j = 1; j < 11; j++) {
      const edibleFilm = [...defaultEdibleFilm];
      edibleFilm[i] += base[i] * (j * 0.05);
      data.push(edibleFilm);
    }
  }
  return data;
}
