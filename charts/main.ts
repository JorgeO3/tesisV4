import { makeInference } from "./model_inference.ts";
import { makeAnalysis } from "./bibliometric_analysis.ts";

const ff = {
  inference: false,
  bibliometricAnalysis: true,
};

if (import.meta.main) {
  ff.inference && makeInference();
  ff.bibliometricAnalysis && makeAnalysis();
}
