import { makeInference } from "./model_inference.ts";
import { BibiometricAnalysis } from "./bibliometric_analysis.ts";
import { Config } from "./analyses_config.ts";

if (import.meta.main) {
  const confg = new Config();
  const ba = new BibiometricAnalysis(confg);

  // FF for disable or enable diferent process
  const { bibliom, inference } = confg.envs;

  // TODO: convert the inferences functions in a class
  inference && makeInference();
  bibliom && ba.start();
}
