if (import.meta.main) {
  const dataPath = new URL("data.csv", import.meta.url).pathname;
  const countriesPath = new URL("countries.csv", import.meta.url).pathname;
  const rawFile = Deno.readTextFileSync(dataPath);
  const authors = parse(rawFile);

  const citedCountries = new Map<string, number>();

  for (let i = 0; i < authors.length; i++) {
    const [_a, _b, _c, _d, _e, citedBy, _g, addres] = authors[i];

    for (let j = 0; j < countries.length; j++) {
      if (addres.includes(countries[j])) {
        const cites = parseInt(citedBy);
        const count = citedCountries.get(countries[j]) || 0;
        citedCountries.set(countries[j], count + cites);
        break;
      }
    }
  }

  await Deno.writeTextFile(countriesPath, "Country, Cites");

  for (const [country, cites] of citedCountries) {
    await Deno.writeTextFile(countriesPath, `\n${country}, ${cites}`, {
      append: true,
    });
  }
}
