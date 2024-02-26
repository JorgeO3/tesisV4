import { either as E, function as fp } from "npm:fp-ts@^2.16.1";
import { assert } from "https://deno.land/std@0.206.0/assert/mod.ts";

const double = (n: number): number => n * 2;

export const imperative = (as: ReadonlyArray<number>): string => {
  const head = (as: ReadonlyArray<number>): number => {
    if (as.length === 0) {
      throw new Error("empty array");
    }
    return as[0];
  };
  const inverse = (n: number): number => {
    if (n === 0) {
      throw new Error("cannot divide by zero");
    }
    return 1 / n;
  };
  try {
    return `Result is ${inverse(double(head(as)))}`;
    // deno-lint-ignore no-explicit-any
  } catch (err: any) {
    return `Error is ${err.message}`;
  }
};

export const functional = (as: ReadonlyArray<number>): string => {
  const head = <A>(as: ReadonlyArray<A>): E.Either<string, A> =>
    as.length === 0 ? E.left("empty array") : E.right(as[0]);
  const inverse = (
    n: number,
  ): E.Either<
    string,
    number
  > => (n === 0 ? E.left("cannot divide by zero") : E.right(1 / n));
  return fp.pipe(
    as,
    head,
    E.map(double),
    E.flatMap(inverse),
    E.match(
      (err) => `Error is ${err}`, // onLeft handler
      (head) => `Result is ${head}`, // onRight handler
    ),
  );
};

console.log(imperative([1, 2, 3]), functional([1, 2, 3]));
console.log(imperative([]), functional([]));
console.log(imperative([0]), functional([0]));
