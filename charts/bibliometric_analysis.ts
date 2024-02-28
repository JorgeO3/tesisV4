import { parse } from "https://deno.land/std@0.217.0/csv/mod.ts";

import { Config } from "./analyses_config.ts";
// import { COUNTRIES } from "./countries.ts";

interface Author {
  firstName: string;
  lastName: string;
  id: string;
}

interface Document {
  authors: Author[];
  title: string;
  year: number;
  citedBy: number;
  countries: string[];
}

interface Documents {
  data: Document[];
  length: number;
}

interface DocsByYear {
  year: number;
  numOfDocs: number;
}

interface DocByAuthor {
  id: number;
  numOfDocs: number;
}

const trimString = (value: string): string => value.trim();

const parseAuthorFullName = (authorInfo: string): Author => {
  const [fullNames, idPart] = authorInfo.split(" (");
  const id = trimString(idPart.replace(")", ""));
  const [lastName, firstName = ""] = fullNames.split(",").map(trimString);

  return { id, lastName, firstName };
};

const parseAuthorFullNames = (row: string): Author[] => {
  return row.split(";").map(parseAuthorFullName);
};

const parseCountries = (affiliations: string): string[] => {
  return affiliations.split(";").map((affiliation) =>
    affiliation.split(",").pop()!.trim()
  );
};
class BibiometricAnalysis {
  docs: Documents;

  constructor(private config: Config) {
    this.docs = this.readAndParseFile();
  }

  private readAndParseFile(): Documents {
    const authorsPath = this.config.authorsFile;
    const file = Deno.readTextFileSync(authorsPath);
    const rawDocuments = parse(file);

    const docsData = rawDocuments
      .slice(1)
      .map(this.parseDoc)
      .filter((doc) => doc.year < 2024);

    return { data: docsData, length: docsData.length };
  }

  // row: Authors [0], Author full names [1], Author(s) ID [2], Title [3], Year [4], Cited by [5], Link [6], Affiliations [7], Authors with affiliations [8], EID [9]
  private parseDoc(row: string[]): Document {
    const authors = parseAuthorFullNames(row[1]);
    const title = row[3];
    const year = parseInt(row[4]);
    const citedBy = parseInt(row[5]);
    const countries = parseCountries(row[7]);

    return {
      authors,
      title,
      year,
      citedBy,
      countries,
    };
  }

  public start(): void {
    // =============== Number of documents by year ===============
    this.numOfDocumentsByYear();

    // =============== Number of documents by Author ===============
    const authors = this.numOfDocumentsByAuthor();

    // =============== Number of cites by country ===============
    this.numOfCitesByCountrie();

    // =============== Number of docs by country ===============
    this.numOfDocsByCountry();

    this.indexH(authors);
  }

  private numOfDocumentsByYear(): DocsByYear[] {
    const { data } = this.docs;
    const docsByYear: Map<number, number> = new Map();

    for (const { year } of data) {
      const docsNum = docsByYear.get(year) || 0;
      docsByYear.set(year, docsNum + 1);
    }

    const orderedData: DocsByYear[] = [];

    for (const [year, numOfDocs] of docsByYear) {
      orderedData.push({ year, numOfDocs });
    }

    orderedData.sort((a, b) => b.year - a.year);

    return orderedData;
  }

  private numOfDocumentsByAuthor(): DocByAuthor[] {
    const { data } = this.docs;
    const docsByAuthor: Map<number, number> = new Map();

    for (const doc of data) {
      for (const { id } of doc.authors) {
        const parsedId = parseInt(id);
        const value = docsByAuthor.get(parsedId) || 0;
        docsByAuthor.set(parsedId, value + 1);
      }
    }

    const orderedDocsByAuthor: DocByAuthor[] = [];

    for (const [id, numOfDocs] of docsByAuthor) {
      orderedDocsByAuthor.push({ id, numOfDocs });
    }
    orderedDocsByAuthor.sort((a, b) => b.numOfDocs - a.numOfDocs);

    return orderedDocsByAuthor;
  }

  private numOfCitesByCountrie() {
    const { data } = this.docs;
    const citesByCountry = new Map<string, number>();

    for (const { citedBy, countries } of data) {
      const countriesVicited: Set<string> = new Set();

      for (const country of countries) {
        if (!countriesVicited.has(country)) {
          const currentCites = citesByCountry.get(country) || 0;
          citesByCountry.set(country, currentCites + citedBy);
          countriesVicited.add(country);
        }
      }
    }

    const orderedCitesByCountry = [];
    for (const [country, cites] of citesByCountry) {
      orderedCitesByCountry.push({ country, cites });
    }
    orderedCitesByCountry.sort((a, b) => b.cites - a.cites);

    // console.log({ orderedCitesByCountry });
    return orderedCitesByCountry;
  }

  private numOfDocsByCountry() {
    const { data } = this.docs;
    const docsByCountry: Map<string, number> = new Map();

    for (const { countries } of data) {
      const countriesVicited = new Set<string>();

      for (const country of countries) {
        if (!countriesVicited.has(country)) {
          const docs = docsByCountry.get(country) || 0;
          docsByCountry.set(country, docs + 1);
          countriesVicited.add(country);
        }
      }
    }
  }

  // [x] Numero de citas por author
  // [x] Total de articulos
  // [-] Primer año de publicacion
  private indexH(docsByAuthor: DocByAuthor[]) {
    const { data } = this.docs;
    const mostImportantAuthors = docsByAuthor.slice(0, 10);
    const citesByAuthor = new Map<string, number>();

    // the number of cites by author
    for (const { authors, citedBy } of data) {
      for (const { id } of authors) {
        const cites = citesByAuthor.get(id) || 0;
        citesByAuthor.set(id, cites + citedBy);
      }
    }

    const sortedCitesByAuthor = [];

    for (const [id, cites] of citesByAuthor) {
      sortedCitesByAuthor.push({ id, cites });
    }
    sortedCitesByAuthor.sort((a, b) => b.cites - a.cites);
  }

  private authorCollaborationNetwork() {}
}

export { BibiometricAnalysis };
