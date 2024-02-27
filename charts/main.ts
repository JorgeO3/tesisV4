import { makeInference } from "./model_inference.ts";
import { makeAnalysis } from "./bibliometric_analysis.ts";
import { Config } from "./envs.ts";

if (import.meta.main) {
  const config = new Config();
  const { bibliometricAnalysis, inference } = config.envs;

  inference && makeInference();
  bibliometricAnalysis && makeAnalysis();
}
