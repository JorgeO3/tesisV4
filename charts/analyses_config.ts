const INFERENCE_ENV = "INFERENCE";
const AUTHORS_FILE = "PATH_FILE_AUTHORS";
const BIBLIMETRIC_ENV = "BIBLIOMETRIC_ANALYSIS";

/**
 * Environment variables for the entire project.
 *
 * inference: FF for enable model inference.
 * bibliom: FF for enabling Bibiometric analysis.
 * authorsFile: path for the
 */
interface Envs {
  inference: boolean;
  bibliom: boolean;
  authorsFile: string;
}

class Config implements Envs {
  inference: boolean;
  bibliom: boolean;
  authorsFile: string;

  constructor() {
    this.inference = !!Deno.env.get(INFERENCE_ENV);
    this.bibliom = !!Deno.env.get(BIBLIMETRIC_ENV);
    this.authorsFile = Deno.env.get(AUTHORS_FILE) || "";

    if (this.bibliom && !this.authorsFile) {
      throw new Error("Please provide the autors file");
    }
  }

  public get envs(): Envs {
    return {
      authorsFile: this.authorsFile,
      inference: this.inference,
      bibliom: this.bibliom,
    };
  }
}

export { Config };
