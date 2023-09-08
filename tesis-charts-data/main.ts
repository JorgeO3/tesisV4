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
  const data: number[][] = [];

  for (let i = 0; i < defaultEdibleFilm.length; i++) {
    for (let j = 0; j < 10; j++) {
      const edibleFilm = [...defaultEdibleFilm];
      edibleFilm[i] = j + 1;
      data.push(edibleFilm);
    }
  }

  const outputsTS = await makeRequestToModel("ts", data);
  const response = await outputsTS.json();
  console.log(response);
}

function makeRequestToModel(
  model: string,
  edibleFilm: number[][],
): Promise<Response> {
  const baseUrl = `http://localhost:8000/model/${model}`;
  const response = fetch(baseUrl, {
    method: "POST",
    body: JSON.stringify(edibleFilm),
    headers: {
      "Content-Type": "application/json",
    },
  });
  return response;
}
