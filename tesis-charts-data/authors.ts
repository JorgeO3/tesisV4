import { parse } from "https://deno.land/std@0.200.0/csv/mod.ts";

interface Author {
  fullName: string;
  codeId: number;
  cites: number;
}

function extractFields(document: string[]): Author[] {
  const [, fullNames, , , , citedBy] = document;
  if (!fullNames) return [];

  const names = fullNames.split(";");
  const authors: Author[] = [];

  for (const name of names) {
    const fullNameParts = name.split(" (");
    const fullName = fullNameParts[0].replace(",", "");
    const code = fullNameParts[1].replace(")", "");

    authors.push({
      cites: parseInt(citedBy) || 0,
      codeId: parseInt(code),
      fullName,
    });
  }
  return authors;
}

if (import.meta.main) {
  const dataFilePath = new URL("data.csv", import.meta.url).pathname;
  const authorsFilePath = new URL("authors.csv", import.meta.url).pathname;

  const rawData = Deno.readTextFileSync(dataFilePath);
  const parsedDocuments = parse(rawData);

  // Id, Name, Cites, Number of documents
  const citedAuthors = new Map<number, [string, number, number]>();

  // Skip the first line of csv file (headers)
  for (let i = 1; i < parsedDocuments.length; i++) {
    const document = parsedDocuments[i];
    const authors = extractFields(document);

    for (const author of authors) {
      const { cites, codeId, fullName } = author;
      const [name, count, documents] = citedAuthors.get(codeId) ||
        [fullName, 0, 0];
      citedAuthors.set(codeId, [name, count + cites, documents + 1]);
    }
  }

  await Deno.writeTextFile(authorsFilePath, "Id, Name, Cites, Documents");
  for (const [id, [name, cites, articles]] of citedAuthors) {
    await Deno.writeTextFile(
      authorsFilePath,
      `\n${id}, ${name}, ${cites}, ${articles}`,
      { append: true },
    );
  }
}
