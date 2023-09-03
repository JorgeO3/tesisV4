import { parse } from "https://deno.land/std@0.200.0/csv/mod.ts";

interface Charactheristic {
  name: string;
  id: string;
  cites: number;
  size: number;
  color: string;
  type: string;
}

interface Author {
  fullName: string;
  id: number;
  cites: number;
}

const NODE_PROPERTY: Record<number, [number, string] | undefined> = {
  0: [5000, "ORANGE"],
  1: [4000, "ORANGE"],
  2: [3000, "ORANGE"],
  3: [2000, "ORANGE"],
};

function getNodeProperties(index: number): [number, string] {
  return NODE_PROPERTY[index] || [1000, "RED"];
}

function parseCSV(filePath: string): string[][] {
  const fileContent = Deno.readTextFileSync(filePath);
  return parse(fileContent);
}

function pathToURL(path: string): string {
  return new URL(path, import.meta.url).pathname;
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
      id: parseInt(code),
      fullName,
    });
  }
  return authors;
}

if (import.meta.main) {
  const numberOfAuthors = 15;
  const dataFilePath = pathToURL("data.csv");
  const authorsFilePath = pathToURL("authors.csv");
  const relationsTablePath = pathToURL("edges.csv");
  const charactheristicTablePath = pathToURL("nodes.csv");

  let parsedDocuments = parseCSV(dataFilePath);
  const _documentsFileHeaders = parsedDocuments[0];
  // Skip the first line of csv file (headers)
  parsedDocuments = parsedDocuments.slice(1);

  let parsedAuthors = parseCSV(authorsFilePath);
  const _authorsFileHeaders = parsedAuthors[0];
  // Skip the first line of csv file (headers)
  parsedAuthors = parsedAuthors.slice(1);

  parsedAuthors = parsedAuthors.sort((a, b) => {
    const aCites = parseInt(a[2]);
    const bCites = parseInt(b[2]);
    return bCites > aCites ? 1 : -1;
  });
  parsedAuthors = parsedAuthors.slice(0, numberOfAuthors);

  // Create the charactheristics table CSV
  const charactheristicsTable: Charactheristic[] = [];

  for (let i = 0; i < parsedAuthors.length; i++) {
    const author = parsedAuthors[i];
    const id = author[0];
    const name = author[1].trimStart();
    const cites = parseInt(author[2]);
    const [size, color] = getNodeProperties(i);
    const type = "PERSON";

    charactheristicsTable.push({
      name,
      id,
      cites,
      size,
      color,
      type,
    });
  }

  Deno.writeTextFileSync(
    charactheristicTablePath,
    "Id,Label",
  );

  for (const charact of charactheristicsTable) {
    const { id, name } = charact;
    Deno.writeTextFileSync(charactheristicTablePath, `\n${id},${name}`, {
      append: true,
    });
  }

  // Create the relationship table CSV
  const relationsTable = new Map<number, [number, number]>();
  const ids = charactheristicsTable.map((charact) => parseInt(charact.id));

  for (let i = 1; i < parsedDocuments.length; i++) {
    const document = parsedDocuments[i];
    const authors = extractFields(document);
    const validAuthors = authors.filter((author) => ids.includes(author.id));

    if (!(validAuthors.length > 2)) continue;
    console.log({ validAuthors });

    for (let j = 0; j < validAuthors.length - 1; j++) {
      const { id: currentId } = validAuthors[j];

      for (let k = 1 + j; k < validAuthors.length; k++) {
        const { id: nextId } = validAuthors[k];

        const key = currentId + nextId;

        const exist = relationsTable.get(key);
        if (exist) continue;
        relationsTable.set(key, [currentId, nextId]);
      }
    }
  }

  Deno.writeTextFileSync(relationsTablePath, "source,target,type,weight");
  for (const [_key, [authorId, coauthorId]] of relationsTable) {
    Deno.writeTextFileSync(
      relationsTablePath,
      `\n${authorId},${coauthorId},Undirected,1`,
      {
        append: true,
      },
    );
  }
}
