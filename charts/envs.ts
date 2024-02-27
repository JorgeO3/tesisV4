const inferenceEnv = "INFERENCE";
const biblimetricEnv = "BIBLIOMETRIC_ANALYSIS";

interface Envs {
  inference: boolean;
  bibliometricAnalysis: boolean;
}

class Config implements Envs {
  inference: boolean;
  bibliometricAnalysis: boolean;

  constructor() {
    this.inference = !!Deno.env.get(inferenceEnv);
    this.bibliometricAnalysis = !!Deno.env.get(biblimetricEnv);
  }

  public get envs(): Envs {
    return {
      inference: this.inference,
      bibliometricAnalysis: this.bibliometricAnalysis,
    };
  }
}

export { Config };
