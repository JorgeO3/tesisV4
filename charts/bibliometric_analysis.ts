import { parse } from "https://deno.land/std@0.217.0/csv/mod.ts";

import { Config } from "./analyses_config.ts";

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

const parseAuthors = (authors: string): string[] => {
  return authors.split(";").map((author) => author.trim());
};

const parseAuthorFullNames = (row: string): Author[] => {
  return row.split(";").map((authorsInfo) => {
    const names = authorsInfo.split(",");
    const firstName = names[0].trim() || "";
    const lastNameWithId = names[1] || "";

    const arrLastNameWithId = lastNameWithId.split(" (");
    const lastName = arrLastNameWithId[0].trim() || "";
    let id = arrLastNameWithId[1] || "";
    id = id.replace(")", "");

    return { firstName, lastName, id };
  });
};

const parseCountries = (affiliations: string): string[] => {
  return affiliations.split(";").map((affiliation) => {
    const arrAffiliation = affiliation.split(",");
    const aalength = arrAffiliation.length;
    return arrAffiliation[aalength - 1].trim();
  });
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

    const docsData: Document[] = rawDocuments.slice(1).map(this.parseDoc);

    return {
      data: docsData,
      length: docsData.length,
    };
  }

  // row: Authors [0], Author full names [1], Author(s) ID [2], Title [3], Year [4], Cited by [5], Link [6], Affiliations [7], Authors with affiliations [8], EID [9]
  private parseDoc(row: string[]): Document {
    const authors = parseAuthorFullNames(row[1]);
    const title = row[3];
    const year = parseInt(row[4]);
    const citedBy = parseInt(row[5]);
    const countries = parseCountries(row[7]); // Asumiendo que row[7] es afiliaciones

    return {
      authors,
      title,
      year,
      citedBy,
      countries,
    };
  }

  public start(): void {
    console.log("docs: ", this.docs);
  }

  private numOfDocumentsByYear() {}
  private numOfDocumentsByAuthor() {}
  private numOfCitesByCountrie() {}
}

export { BibiometricAnalysis };
