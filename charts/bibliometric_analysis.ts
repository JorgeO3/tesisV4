// import { parse } from "https://deno.land/std@0.217.0/csv/mod.ts";
// import { Config } from "./analyses_config.ts";

// interface Author {
//   firstName: string;
//   lastName: string;
// }

// interface Document {
//   authors: Author[];
//   title: string;
//   year: number;
//   citedBy: number;
//   countries: string[];
// }

// class BibiometricAnalysis {
//   docs: Document[];
//   authors: Map<string, Author>;

//   constructor() {
//     this.docs = this.readAndParseAuthorsFile();
//     this.authors = this.searchAndSetAuthors();
//   }

//   readAndParseAuthorsFile(): Document[] {}
//   searchAndSetAuthors(): Map<string, Author> {}
// }

// export { BibiometricAnalysis };

import { parse } from "https://deno.land/std@0.217.0/csv/mod.ts";

import { Config } from "./analyses_config.ts";
import { stringify } from "https://deno.land/std@0.217.0/csv/stringify.ts";
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
  private basePath = "./charts";
  private docs: Documents;

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
    this.numOfDocumentsByYear();
    const authors = this.numOfDocumentsByAuthor();
    this.numOfCitesByCountrie();
    this.numOfDocsByCountry();
    this.indexH(authors);
    this.authorCollaborationNetwork(authors);
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

    let csvFile = "Año, Numero de documentos\n";
    for (const { year, numOfDocs } of orderedData) {
      csvFile += `${year}, ${numOfDocs}\n`;
    }

    Deno.writeTextFileSync(`${this.basePath}/documents_by_year.csv`, csvFile, {
      append: false,
    });

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

    let csvFile = "ID Autor, Número de documentos\n";

    for (const { id, numOfDocs } of orderedDocsByAuthor) {
      csvFile += `${id}, ${numOfDocs}\n`;
    }

    Deno.writeTextFileSync(
      `${this.basePath}/documents_by_author.csv`,
      csvFile,
      {
        append: false,
      },
    );

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

    let csvFile = "Pais, Citas\n";

    for (const { country, cites } of orderedCitesByCountry) {
      csvFile += `${country}, ${cites}\n`;
    }

    Deno.writeTextFileSync(`${this.basePath}/cites_by_country.csv`, csvFile, {
      append: false,
    });

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
    const path = `${this.basePath}/bibliometric_indicators.csv`;

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

    const firstPublications = new Map<string, number>();

    // fist year of publication
    for (const { authors, year } of data) {
      const authorsSet = authors.reduce((acc, { id }) => {
        return acc.add(id);
      }, new Set<string>());

      for (const { id } of mostImportantAuthors) {
        const parsedId = id.toString();
        const currentFirstPublication = firstPublications.get(parsedId) || 2024;

        if (authorsSet.has(parsedId) && year < currentFirstPublication) {
          firstPublications.set(parsedId, year);
        }
      }
    }

    const sortedArrFP = [];

    for (const [authorId, year] of firstPublications) {
      sortedArrFP.push({ authorId, year });
    }

    let csvFile =
      "ID, Autor, Índice H, Total de Citas, Total de artículos, Primer año\n";

    for (let i = 0; i < 10; i++) {
      const authorId = sortedArrFP[i].authorId;
      const year = sortedArrFP[i].year;
      const cites = citesByAuthor.get(authorId)!;
      const { numOfDocs } = docsByAuthor.find(({ id }) =>
        id.toString() == authorId
      )!;

      csvFile += `${authorId},,, ${cites}, ${numOfDocs}, ${year}\n`;
    }

    this.saveFile(path, csvFile);
  }

  private authorCollaborationNetwork(docsByAuthor: DocByAuthor[]) {
    const { data } = this.docs;
    const networkByAuthor = new Map<string, Set<string>>();
    const mostImportantAuthors = docsByAuthor
      .slice(0, 20)
      .reduce((acc, { id }) => {
        return acc.add(id.toString());
      }, new Set<string>());

    for (const { authors } of data) {
      const importantAuthorsInCurrentDocument = authors
        .filter((author) => mostImportantAuthors.has(author.id))
        .map((author) => author.id);

      for (const authorId of importantAuthorsInCurrentDocument) {
        const network = networkByAuthor.get(authorId) || new Set();

        for (const relatedAuthorId of importantAuthorsInCurrentDocument) {
          if (authorId == relatedAuthorId) continue;
          network.add(relatedAuthorId);
        }
        networkByAuthor.set(authorId, network);
      }
    }

    // create files for gephi
    const relationTables = new Set<[string, string]>();

    for (const [author, couthors] of networkByAuthor) {
      const couthorsArr = Array.from(couthors);

      for (const coauthor of couthorsArr) {
        relationTables.add([author, coauthor]);
      }
    }

    // Nodes tables
    // TODO: the label is selected manually
    const nodePathFile = `${this.basePath}/nodes.csv`;
    const mostImportantAuthorsArr = Array.from(mostImportantAuthors);
    const authorsByNumOfDocs = docsByAuthor
      .filter((val) => mostImportantAuthorsArr.includes(val.id.toString()));

    let csvNodeFile = "Id,Label,Weight\n";
    const authors = data
      .flatMap((doc) => doc.authors)
      .filter(({ id }) => mostImportantAuthorsArr.includes(id));
    const cleanedAuthors = authors
      .filter((author, i) => authors.indexOf(author) == i);

    for (const { id, numOfDocs } of authorsByNumOfDocs) {
      const weight = this.normalizeNumOfDocs(numOfDocs);
      const { lastName, firstName } = cleanedAuthors
        .find((author) => author.id == id.toString())!;

      csvNodeFile += `${id},${lastName + " " + firstName},${weight}\n`;
    }

    this.saveFile(nodePathFile, csvNodeFile);

    // Edges tables
    let csvEdgeFile = "Source,Target,Type\n";
    const relationTablesArr = Array.from(relationTables);
    const edgePathFile = `${this.basePath}/edges.csv`;

    for (const [author1, author2] of relationTablesArr) {
      csvEdgeFile += `${author1},${author2},Undirected\n`;
    }
    this.saveFile(edgePathFile, csvEdgeFile);
  }

  private normalizeNumOfDocs(numOfDocs: number) {
    const [max, maxLevel] = [29, 3];
    const result = (numOfDocs / max) * maxLevel;
    return Math.floor(result);
  }

  private saveFile(path: string, file: string, append = false) {
    Deno.writeTextFileSync(path, file, { append });
  }
}

export { BibiometricAnalysis };
